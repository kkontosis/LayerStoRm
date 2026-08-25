// Cache, graph, stream, placement, sequence, and config handlers.
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/v4_kv_tiering.h"

#include <cstdio>
#include <cstdlib>
#include "daemon/config_update.h"
#include "daemon/dispatch_detail.h"
#include "daemon/expert_lifecycle_manager.h"

#include <spdlog/spdlog.h>

#include "compute/graphs/graph_registry.h"
#include "compute/stream_manager.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/device_backend.h"        // ticket J: V4 kMain zero-on-claim
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h" // ticket J: V4 entry geometry
#include "core/statistics/coactivation_graph.h"
#include "daemon/kv_tiering_manager.h"  // GLM-25k
#include "daemon/spsc_ring.h"

namespace layerstorm::daemon {

static compute::GraphType to_graph_type(uint8_t t) {
    if (t >= static_cast<uint8_t>(compute::GraphType::kCount))
        return compute::GraphType::kCount;
    return static_cast<compute::GraphType>(t);
}

// ── Cache handlers ─────────────────────────────────────────────────────────

void CommandDispatcher::handle_cache_reserve(const ipc::Command& cmd) {
    const auto& p = cmd.cache_reserve;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    void* addr = deps_.expert_cache->reserve(
        key, gpu, to_zone(p.zone), p.is_duplicate != 0);

    if (!addr) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kCacheReserveFailed, "cache reserve failed: zone full");
        return;
    }
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
}

void CommandDispatcher::handle_cache_evict(const ipc::Command& cmd) {
    const auto& p = cmd.cache_op;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    // TD-EVICT-BOARD-DESYNC: ExpertCache::evict fires the residency listener.
    bool ok = deps_.expert_cache->evict(key, gpu);
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
}

void CommandDispatcher::handle_cache_promote(const ipc::Command& cmd) {
    const auto& p = cmd.cache_op;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    bool ok = deps_.elm ? deps_.elm->promote(key, gpu)
                        : deps_.expert_cache->promote(key, gpu);
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
}

void CommandDispatcher::handle_cache_demote(const ipc::Command& cmd) {
    const auto& p = cmd.cache_op;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    bool ok = deps_.elm ? deps_.elm->demote(key, gpu)
                        : deps_.expert_cache->demote(key, gpu);
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
}

// ── Graph replay ───────────────────────────────────────────────────────────

void CommandDispatcher::handle_graph_replay(const ipc::Command& cmd) {
    auto gtype = to_graph_type(cmd.graph.graph_type);
    if (gtype == compute::GraphType::kCount) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kGraphError, "invalid graph type");
        return;
    }

    compute::GraphKey gkey{
        gtype,
        static_cast<int>(cmd.gpu_idx),
        static_cast<int>(cmd.graph.batch_size)};

    // TODO:DEBT TD-37: graph_registry dereferenced without null check
    auto* entry = deps_.graph_registry->find(gkey);
    if (!entry) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kGraphError, "graph not captured");
        return;
    }

    // Graph entry found — enqueue async completion via event.
    const int gpu = static_cast<int>(cmd.gpu_idx);
    void* event = create_and_record_event(gpu, to_stream(cmd.stream_id));

    PendingCompute pc{};
    pc.cmd_seq    = cmd.cmd_seq;
    pc.gpu_idx    = cmd.gpu_idx;
    pc.cmd_type   = cmd.cmd_type;
    pc.layer_idx  = 0;
    pc.cuda_event = event;
    pending_compute_.push_back(pc);
}

// ── Stream event handlers ──────────────────────────────────────────────────

void CommandDispatcher::handle_record_event(const ipc::Command& cmd) {
    uint32_t eid = cmd.event.event_id;

    // Lazy-create event if not in pool.
    const int gpu = static_cast<int>(cmd.gpu_idx);
    auto it = event_pool_.find(eid);
    if (it == event_pool_.end()) {
        void* ev = deps_.stream_manager->create_event(gpu);
        it = event_pool_.emplace(eid, PooledEvent{ev, gpu}).first;
    }

    deps_.stream_manager->record_event(
        it->second.handle,
        static_cast<int>(cmd.gpu_idx),
        to_stream(cmd.stream_id));

    write_event_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
}

void CommandDispatcher::handle_stream_wait_event(const ipc::Command& cmd) {
    uint32_t eid = cmd.event.event_id;

    auto it = event_pool_.find(eid);
    if (it == event_pool_.end()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kEventPoolError,
                    "stream_wait: unknown event_id");
        return;
    }

    deps_.stream_manager->wait_event(
        static_cast<int>(cmd.gpu_idx),
        to_stream(cmd.stream_id),
        it->second.handle);

    write_event_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
}

// ── Placement / NUMA handlers ──────────────────────────────────────────────

void CommandDispatcher::handle_compute_affinity_hints(const ipc::Command& cmd) {
    const auto& p = cmd.affinity_hints;
    auto num_gpus = p.num_gpus;

    // Build a span over the capacity slots.
    // gpu_capacity_slots is uint32_t[] but compute_affinity_hints expects
    // span<const int>.  Cast is safe for same-width types.
    static_assert(sizeof(int) == sizeof(uint32_t));
    std::span<const int> caps(
        reinterpret_cast<const int*>(p.gpu_capacity_slots),
        static_cast<size_t>(num_gpus));

    auto hints = deps_.coactivation_graph->compute_affinity_hints(num_gpus, caps);

    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
}

void CommandDispatcher::handle_numa_migrate(const ipc::Command& cmd) {
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNumaMissing,
                    "NUMA migrate: NVMe tier not available");
        return;
    }

    const auto& p = cmd.numa_migrate;
    auto key = make_key(p.layer_idx, p.expert_idx);

    // WP-5: mmap-backed NvmeTier — pages are OS-managed, NUMA migration
    // not applicable.  Report as no-op success.
    (void)key;
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0u);
}

// ── Sequence lifecycle handlers (IPC-8a) ──────────────────────────────────

void CommandDispatcher::handle_seq_create(const ipc::Command& cmd) {
    if (!deps_.page_allocator) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqCreate,
                    "seq_create: page allocator not configured");
        return;
    }

    const auto& p = cmd.seq_create;
    const uint64_t seq_id = p.seq_id;
    const uint32_t prompt_len = p.prompt_len;
    const auto pool = (p.pool == 1) ? memory::Pool::kSpeculation
                                    : memory::Pool::kMain;

    if (sequences_.count(seq_id)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqCreate,
                    "seq_create: seq_id already exists");
        return;
    }

    auto* pa = deps_.page_allocator;
    const auto& dcp = pa->dcp_config();
    // TD-V4-KMAIN-SIZING: V4 page arithmetic must use the dispatcher's
    // authoritative page granularity (deps_.kv_page_size = logical_block_
    // tokens, 256 native tokens per kMain CSA page), NOT the DcpConfig
    // value: at tp == 1 set_dcp_config is never called and the default (16)
    // overcounted V4 logical pages 16×, exhausting the CSA bucket at
    // prompt_len ≈ 4000 while ensure_pages (which already used
    // deps_.kv_page_size) grew correctly. Non-V4 keeps the DcpConfig value
    // (the engine sets both to memory.kv_cache.page_size_tokens; unit
    // fixtures rely on the DcpConfig one).
    const bool is_v4_arch = deps_.live_config
        && deps_.live_config->model.architecture
               == config::Architecture::deepseek_v4;
    const auto page_size = static_cast<uint32_t>(
        (is_v4_arch && deps_.kv_page_size > 0) ? deps_.kv_page_size
                                               : dcp.page_size_tokens);
    // KD-4e1: pre-allocate headroom pages beyond prompt_len.
    const uint32_t base_pages = (prompt_len + page_size - 1) / page_size;
    const uint32_t with_headroom = base_pages
        + static_cast<uint32_t>(chunk_size_pages_);
    uint32_t num_pages = (max_blocks_per_seq_ > 0)
        ? std::min(with_headroom,
                   static_cast<uint32_t>(max_blocks_per_seq_))
        : with_headroom;

    // TD-KVT-ADMISSION-UPFRONT (memory.kv_tiering.tiered_prefill): windowed
    // admission — allocate only the admission window upfront; the remainder
    // grows lazily (ensure_pages at kv-meta build) as the chunk frontier
    // advances while chunk-boundary demotion frees behind the retention
    // window, so peak KV VRAM is bounded by the window, never prompt
    // length.  kMain only (draft pools stay upfront), non-V4 (three-bucket
    // layout keeps full allocation — TD note), and only with the tiering
    // manager LIVE (without demotion a window would just move the
    // exhaustion mid-prefill).
    if (pool == memory::Pool::kMain && kv_tiering_ && !is_v4_arch
        && deps_.live_config
        && deps_.live_config->memory.kv_tiering.tiered_prefill
        && deps_.live_config->compute.dsa_sparse_prefill) {
        const int cfg_w =
            deps_.live_config->memory.kv_tiering.admission_window_tokens;
        const uint32_t w_tokens = cfg_w > 0
            ? static_cast<uint32_t>(cfg_w)
            : static_cast<uint32_t>(kv_tiering_->hot_buffer_slots()) + 1024u;
        const uint32_t w_pages = (w_tokens + page_size - 1) / page_size
            + static_cast<uint32_t>(chunk_size_pages_);
        if (w_pages < num_pages) {
            spdlog::info(
                "seq_create: windowed admission for seq {} — prompt_len {} "
                "({} logical pages) admitted with a {}-token window ({} "
                "pages upfront); remainder grows lazily, demotion bounds "
                "the hot set (TD-KVT-ADMISSION-UPFRONT)",
                seq_id, prompt_len, num_pages, w_tokens, w_pages);
            num_pages = w_pages;
        }
    }

    // TD-GOLDEN: one physical page per (logical page, layer), layer-major —
    // every attention layer needs its own KV rows for a token.
    const int kv_layers = kv_layers_ > 0 ? kv_layers_ : 1;
    std::vector<memory::PageHandle> handles;
    handles.reserve(static_cast<size_t>(num_pages) * kv_layers);

    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t token_start = i * page_size;
        const uint32_t token_end = token_start + page_size;

        for (int l = 0; l < kv_layers; ++l) {
            // TD-V4-KMAIN-SIZING: V4 non-CSA layers keep their KV in the
            // side pools — sentinel slot, no kMain/spec page consumed.
            if (!layer_takes_kmain_page(l)) {
                memory::PageHandle s;
                s.gpu_idx = static_cast<int>(cmd.gpu_idx);
                s.page_idx = -1;
                s.gpu_ptr = nullptr;
                s.pool = pool;
                handles.push_back(s);
                continue;
            }
            std::optional<memory::PageHandle> h;
            if (dcp.enabled() && pool == memory::Pool::kMain) {
                h = pa->allocate_for_dcp_append(
                    seq_id, token_start, static_cast<uint32_t>(l),
                    static_cast<int>(cmd.gpu_idx));
            } else if (dcp.enabled() && !dcp.kv_sharded
                       && pool == memory::Pool::kSpeculation) {
                // INV-KV-REP extended to kSpeculation (TD-KV-REPLICATED-SPEC):
                // build_kv_metadata replicates the draft's page_idx to every
                // rank's block table, so the page must be claimed at the SAME
                // index on EVERY TP GPU in lockstep — single-GPU allocation
                // here would alias physical pages exactly like the kMain
                // TD-KV-REPLICATED-PAGE-ALIAS bug.  Sharded KV keeps the
                // owner-only single-GPU path below.
                h = pa->allocate_replicated(seq_id, token_start,
                                            static_cast<uint32_t>(l),
                                            memory::Pool::kSpeculation);
            } else {
                h = pa->allocate(static_cast<int>(cmd.gpu_idx), pool);
                if (h) {
                    auto& m = pa->meta(*h);
                    m.sequence_id = seq_id;
                    m.token_start = token_start;
                    m.token_end   = token_end;
                    m.layer_index = static_cast<uint32_t>(l);
                }
            }

            if (!h) {
                // Rollback: free everything allocated so far (skip
                // sentinels — no VRAM behind them).
                size_t real_allocated = 0;
                for (auto& prev : handles) {
                    if (prev.page_idx >= 0) {
                        pa->free(prev);
                        ++real_allocated;
                    }
                }
                // Pool identity + counts: which pool ran dry, how much the
                // request needs vs what the pool holds (first-failure
                // diagnosis instead of a bare "exhausted").
                const char* pool_name =
                    pool == memory::Pool::kSpeculation ? "kSpeculation"
                                                       : "kMain";
                const int gpu_q = static_cast<int>(cmd.gpu_idx);
                const int pool_free = pa->free_pages(gpu_q, pool);
                const int pool_total = pa->total_pages(gpu_q, pool);
                int real_layers = 0;
                for (int ll = 0; ll < kv_layers; ++ll)
                    if (layer_takes_kmain_page(ll)) ++real_layers;
                const std::string msg =
                    std::string("seq_create: ") + pool_name
                    + " pool exhausted (need "
                    + std::to_string(
                          static_cast<uint64_t>(num_pages) * real_layers)
                    + " pages = " + std::to_string(num_pages) + " logical x "
                    + std::to_string(real_layers) + " layers, got "
                    + std::to_string(real_allocated) + ", pool "
                    + std::to_string(pool_free) + "/"
                    + std::to_string(pool_total) + " free gpu"
                    + std::to_string(gpu_q) + ")";
                spdlog::error(
                    "seq_create failed: seq {} prompt_len {} page_size {} "
                    "-> {} logical pages x {} kMain layers (of {} kv layers)"
                    " from {}; allocated {} before exhaustion; pool free "
                    "{}/{} on gpu {}",
                    seq_id, prompt_len, page_size, num_pages, real_layers,
                    kv_layers, pool_name, real_allocated, pool_free,
                    pool_total, gpu_q);
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kSeqCreate, msg.c_str());
                return;
            }
            handles.push_back(*h);
        }
    }

    // Ticket J determinism: V4's CSA tier lives in kMain pages whose entries
    // are written sparsely (one 1160-B entry per completed 4-token block) —
    // a page re-claimed from a finished sequence carries the previous
    // holder's residue in every not-yet-written entry. Zero V4 kMain/spec
    // pages at claim so a fresh sequence starts from the state the ticket-H
    // goldens validated (the side tiers already get this in
    // ensure_v4_tier_pages).
    if (deps_.live_config
        && deps_.live_config->model.architecture ==
               config::Architecture::deepseek_v4
        && deps_.stream_manager) {
        // V4-5T: CSA entry bytes come from the allocator's layout (the
        // codec-axis authority — 1160 FP8 / 644 TQ). A format-blind FP8
        // constant here would OVERRUN TQ-sized kMain pages into their
        // neighbors (page bytes = entries × entry_bytes).
        const int64_t csa_entry_bytes =
            deps_.page_allocator->v4_layout().enabled
                ? deps_.page_allocator->v4_layout().csa_entry_bytes
                : memory::kV4Fp8EntryBytes;
        const int64_t page_bytes =
            static_cast<int64_t>(page_size) / memory::kV4CsaRatio
            * csa_entry_bytes;
        for (const auto& h : handles) {
            if (!h.gpu_ptr || h.gpu_idx < 0
                || static_cast<size_t>(h.gpu_idx)
                       >= deps_.device_backends.size()
                || !deps_.device_backends[static_cast<size_t>(h.gpu_idx)])
                continue;
            auto* be = deps_.device_backends[static_cast<size_t>(h.gpu_idx)];
            be->set_device();
            be->memset_async(h.gpu_ptr, 0, static_cast<size_t>(page_bytes),
                             deps_.stream_manager->stream(
                                 h.gpu_idx, compute::StreamId::kAttention));
        }
    }

    const auto allocated = static_cast<uint32_t>(handles.size());
    // INV-SEQ-FORK-STATE enumeration point (create): the structured binding
    // is the compile-time completeness gate — adding a SequenceState member
    // breaks it, forcing an explicit init decision here (and clone-vs-
    // exclude at fork, release at free).
    {
        SequenceState st;
        auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers] = st;
        kv_pages = std::move(handles);
        forked = false;         // fresh create
        (void)indexer_pages;    // provisioned lazily (ensure_indexer_pages)
        (void)indexer_cov;      // untracked until the first attention step
        (void)v4_tiers;         // provisioned lazily (ensure_v4_tier_pages)
        sequences_.emplace(seq_id, std::move(st));
    }
    invalidate_kv_meta();  // invalidate dirty guard
    ++page_budget_.active_sequences;

    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, allocated, 0);
}

void CommandDispatcher::handle_seq_free(const ipc::Command& cmd) {
    if (!deps_.page_allocator) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFree,
                    "seq_free: page allocator not configured");
        return;
    }

    const uint64_t seq_id = cmd.seq_free.seq_id;

    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFree,
                    "seq_free: unknown seq_id");
        return;
    }

    // INV-SEQ-FORK-STATE enumeration point (free): the structured binding is
    // the compile-time completeness gate — adding a SequenceState member
    // breaks it, forcing an explicit release decision here.
    auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers] =
        it->second;
    (void)indexer_cov;  // no external resources — dies with the aggregate

    // GLM-25k: drain in-flight demotions + drop tiering state BEFORE the bulk
    // free below (demoted handles were already freed by the manager and are
    // neutralized to page_idx == -1 — skip them to avoid a double free).
    if (kv_tiering_) kv_tiering_->on_seq_free(seq_id);

    const auto freed = static_cast<uint32_t>(kv_pages.size());
    for (auto& h : kv_pages) {
        if (h.page_idx < 0) continue;  // GLM-25k: demoted (already freed)
        deps_.page_allocator->free(h);
    }
    // TD-GLM-INDEXER-PAGED: return the sequence's indexer-K pool pages and
    // clear the host page table (stale device pointers must not survive into
    // the next sequence's coverage checks; live rows are rewritten by
    // ensure_indexer_pages before every executor read).
    if (!indexer_pages.empty()) {
        for (auto& h : indexer_pages) deps_.page_allocator->free(h);
        for (auto& table : indexer_page_table_)
            std::fill(table.begin(), table.end(), nullptr);
    }
    // V4-7b: return the sequence's side-tier pages (kSwa ring, kHca, LID)
    // and drop the executor-side compressor state rings. swa.empty() ⇔ the
    // tiers were never provisioned (ensure_v4_tier_pages resizes swa first).
    if (!v4_tiers.swa.empty()) {
        auto free_h = [&](const memory::PageHandle& h) {
            if (h.page_idx >= 0 && h.gpu_ptr) deps_.page_allocator->free(h);
        };
        for (auto& rv : v4_tiers.swa) for (auto& h : rv) free_h(h);
        for (auto& rv : v4_tiers.hca)
            for (auto& v : rv) for (auto& h : v) free_h(h);
        for (auto& rv : v4_tiers.lid)
            for (auto& v : rv) for (auto& h : v) free_h(h);
        if (deps_.dcp_executor) deps_.dcp_executor->v4_free_sequence(seq_id);
    }
    if (v4_kv_tiering_) v4_kv_tiering_->free_sequence(seq_id);
    const bool was_fork = forked;
    sequences_.erase(it);
    invalidate_kv_meta();  // invalidate dirty guard
    if (page_budget_.active_sequences > 0) --page_budget_.active_sequences;
    if (was_fork && page_budget_.active_forks > 0) --page_budget_.active_forks;

    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, freed, 0);
}

void CommandDispatcher::handle_seq_fork(const ipc::Command& cmd) {
    if (!deps_.page_allocator) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFork,
                    "seq_fork: page allocator not configured");
        return;
    }

    const uint64_t src_id = cmd.seq_fork.src_seq_id;
    const uint64_t dst_id = cmd.seq_fork.dst_seq_id;

    auto src_it = sequences_.find(src_id);
    if (src_it == sequences_.end()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFork,
                    "seq_fork: unknown source seq_id");
        return;
    }

    if (sequences_.count(dst_id)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFork,
                    "seq_fork: destination seq_id already exists");
        return;
    }

    // INV-SEQ-FORK-STATE enumeration point (fork): CMD_SEQ_FORK must clone
    // EVERY piece of per-sequence engine state, not just KV pages — the
    // structured binding below is the compile-time completeness gate: adding
    // a SequenceState member breaks it, forcing an explicit clone-vs-
    // exclude-with-justification decision HERE (the old five-parallel-maps
    // scheme silently missed indexer_cov — TD-PREFIX-FORK-COV: forked
    // children decoded permanently DENSE). External per-seq families the
    // fork must also handle: kv_tiering_ (on_seq_fork below),
    // v4_kv_tiering_ (cold-page deep copy), the DcpExecutor V4 state rings
    // (v4_fork_sequence), and the DSpark draft context (engine-side
    // adoption — advisory, acceptance-only).
    const auto& [src_kv_pages, src_forked, src_indexer_pages, src_cov,
                 src_v4_tiers] = src_it->second;
    (void)src_forked;  // dst is a fork regardless of the parent's origin

    // GLM-25k (TD-KVT-SPEC-FORK lifted): forking a demoted parent.  The
    // tiering manager drains the parent's in-flight demotions, then the
    // child REFCOUNT-shares the parent's cold slots (no cold-byte copy; a
    // slot returns to the pool only when its last holder releases it) and
    // inherits the page states / demoted frontier.  The parent's demoted
    // handles are neutralized (page_idx = -1, VRAM freed) — they are copied
    // as-is below (no add_ref on a freed page): the child's KV for those
    // pages lives in the shared pinned cold pool, and a child rewind into
    // that territory re-promotes the child's OWN fresh VRAM copies
    // (copy-on-write split, KvTieringManager::repromote_seq).  Drafts /
    // graph replay on the demoted family stay fail-closed (Phase-12).
    if (kv_tiering_) kv_tiering_->on_seq_fork(src_id, dst_id);

    const auto& src_handles = src_kv_pages;
    std::vector<memory::PageHandle> dst_handles;
    dst_handles.reserve(src_handles.size());

    for (const auto& h : src_handles) {
        if (h.page_idx >= 0)  // demoted handles shared via cold-slot refcounts
            deps_.page_allocator->add_ref(h);
        dst_handles.push_back(h);
    }

    // TD-51ci: CoW-copy the last (partial) LOGICAL page so the child can
    // write new tokens without corrupting the parent's KV data.  The
    // parent's position is frozen after fork (speculation lifecycle), so
    // only the partial page needs splitting — earlier pages are read-only.
    // TD-GOLDEN/INV-KV-LAYER: a logical page is kv_layers_ physical pages
    // (layer-major) — ALL of them must be split, not just the last handle
    // (which is only layer L-1; sharing the rest would let the child's
    // k_append corrupt the parent's KV for layers 0..L-2).
    const size_t L = static_cast<size_t>(kv_layers_ > 0 ? kv_layers_ : 1);
    if (dst_handles.size() >= L) {
        const size_t first = dst_handles.size() - L;
        for (size_t i = first; i < dst_handles.size(); ++i) {
            auto& h = dst_handles[i];
            // GLM-25k: the append-frontier (last logical) page never demotes
            // (INV-KVT-4), so these handles are always valid; guard anyway —
            // cow_copy on a neutralized handle would corrupt the allocator.
            if (h.page_idx < 0) continue;
            try {
                h = deps_.page_allocator->cow_copy(h);
                // Update metadata: child owns this page.
                auto& m = deps_.page_allocator->meta(h);
                m.sequence_id = dst_id;
            } catch (const std::runtime_error&) {
                // Pool exhausted — roll back entire fork (demoted handles
                // hold no VRAM ref; return the child's cold-slot refs).
                for (auto& hh : dst_handles)
                    if (hh.page_idx >= 0) deps_.page_allocator->free(hh);
                if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kSeqFork,
                            "seq_fork: pool exhausted during CoW split");
                return;
            }
        }
    }

    // DSA models (TD-GLM-INDEXER-PAGED): the indexer K-cache is a SEPARATE
    // per-seq page table that must fork WITH the KV — without it a forked
    // child has no indexer K for the prefix and sparse decode is broken
    // (prefix-cache blocker found 2026-08-18). Same pattern as the KV
    // pages: add_ref every handle, CoW the append-frontier logical page
    // GROUP (all (computing-layer, rank) handles of the last page —
    // handles are (page, layer, rank)-ordered) so the child's indexer
    // appends cannot corrupt the parent's.
    std::vector<memory::PageHandle> dst_ik;
    if (!src_indexer_pages.empty()) {
        const auto& src_ik = src_indexer_pages;
        int n_computing = 0;
        for (uint8_t c : indexer_computes_) n_computing += c;
        const int dcp = deps_.dcp_executor ? deps_.dcp_executor->dcp_size()
                                           : 1;
        const bool ik_local = dcp >= 2 && deps_.live_config
            && deps_.live_config->hardware.dcp_indexer_mode
                   == config::DcpIndexerMode::local;
        const size_t group = static_cast<size_t>(
            std::max(1, n_computing) * (ik_local ? 1 : std::max(1, dcp)));
        dst_ik.reserve(src_ik.size());
        for (const auto& h : src_ik) {
            deps_.page_allocator->add_ref(h);
            dst_ik.push_back(h);
        }
        auto rollback_all = [&]() {
            for (auto& hh : dst_ik) deps_.page_allocator->free(hh);
            for (auto& hh : dst_handles)
                if (hh.page_idx >= 0) deps_.page_allocator->free(hh);
            if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
        };
        if (dst_ik.size() >= group) {
            const size_t first = dst_ik.size() - group;
            for (size_t i = first; i < dst_ik.size(); ++i) {
                try {
                    dst_ik[i] = deps_.page_allocator->cow_copy(dst_ik[i]);
                    deps_.page_allocator->meta(dst_ik[i]).sequence_id =
                        dst_id;
                } catch (const std::runtime_error&) {
                    rollback_all();
                    write_error(cmd.cmd_seq, cmd.gpu_idx,
                                ipc::CmpErrorCategory::kSeqFork,
                                "seq_fork: indexer-K pool exhausted during "
                                "CoW split");
                    return;
                }
            }
        }
    }

    // TD-V4-SERVE-PREFIX (2026-08-22): V4 side tiers are COPY-ON-FORK.
    // Unlike kMain pages (refcount + frontier CoW), the side tiers are
    // mutated IN PLACE by every append — the SWA ring overwrites slot
    // pos % window, the compressor state rings cycle pos % capacity, and
    // HCA/LID entries land in the last partially-filled page — so a child
    // sharing the parent's pages would corrupt the parent's history on its
    // first decode step. The tier byte volume is small (~MBs vs the kMain
    // GBs), so the child gets FRESH pages with a D2D copy of the parent's
    // content (per rank, per layer, on each GPU's kAttention stream —
    // ordered after the parent's last tier writes on the same stream),
    // and the executor clones the per-seq state rings + step tracking
    // (v4_fork_sequence) so the child resumes at the parent's frontier.
    V4SeqTiers dst_tiers;
    const bool v4_fork = !src_v4_tiers.swa.empty();
    auto free_tier_pages = [&]() {
        auto free_h = [&](const memory::PageHandle& h) {
            if (h.page_idx >= 0 && h.gpu_ptr) deps_.page_allocator->free(h);
        };
        for (auto& rv : dst_tiers.swa) for (auto& h : rv) free_h(h);
        for (auto& rv : dst_tiers.hca)
            for (auto& v : rv) for (auto& h : v) free_h(h);
        for (auto& rv : dst_tiers.lid)
            for (auto& v : rv) for (auto& h : v) free_h(h);
    };
    if (v4_fork) {
        const auto& cfg = *deps_.live_config;
        constexpr int64_t kV4EntryBytes = 1160;   // deps V4CacheLayout
        constexpr int64_t kV4LidEntryBytes = 132; // 128 FP8 + f32 scale
        const int PT = cfg.memory.kv_cache.indexer_k_page_size_tokens;
        const int64_t swa_bytes =
            static_cast<int64_t>(cfg.model.sliding_window) * kV4EntryBytes;
        const int64_t hca_bytes =
            static_cast<int64_t>(memory::kV4LogicalBlockTokens
                                 / memory::kV4HcaRatio) * kV4EntryBytes;
        const int64_t lid_bytes =
            static_cast<int64_t>(PT / memory::kV4CsaRatio) * kV4LidEntryBytes;
        auto copy_page = [&](const memory::PageHandle& srch, int64_t bytes,
                             memory::Pool pool)
                -> std::optional<memory::PageHandle> {
            auto h = deps_.page_allocator->allocate(srch.gpu_idx, pool);
            if (!h || !h->gpu_ptr) return std::nullopt;
            auto& m = deps_.page_allocator->meta(*h);
            m.sequence_id = dst_id;
            m.layer_index =
                deps_.page_allocator->meta(srch).layer_index;
            const auto g = static_cast<size_t>(srch.gpu_idx);
            if (g < deps_.device_backends.size()
                && deps_.device_backends[g] && deps_.stream_manager) {
                auto* be = deps_.device_backends[g];
                be->set_device();
                be->memcpy_d2d_async(h->gpu_ptr, srch.gpu_ptr,
                                     static_cast<size_t>(bytes),
                                     deps_.stream_manager->stream(
                                         srch.gpu_idx,
                                         compute::StreamId::kAttention));
            }
            return h;
        };
        bool tier_ok = true;
        const size_t n_ranks = src_v4_tiers.swa.size();
        dst_tiers.swa.resize(n_ranks);
        dst_tiers.hca.resize(n_ranks);
        dst_tiers.lid.resize(n_ranks);
        for (size_t r = 0; r < n_ranks && tier_ok; ++r) {
            dst_tiers.swa[r].resize(src_v4_tiers.swa[r].size());
            for (size_t l = 0; l < src_v4_tiers.swa[r].size() && tier_ok;
                 ++l) {
                const auto& sh = src_v4_tiers.swa[r][l];
                if (sh.page_idx < 0 || !sh.gpu_ptr) continue;
                auto h = copy_page(sh, swa_bytes, memory::Pool::kSwa);
                if (!h) { tier_ok = false; break; }
                dst_tiers.swa[r][l] = *h;
            }
            dst_tiers.hca[r].resize(src_v4_tiers.hca[r].size());
            for (size_t l = 0; l < src_v4_tiers.hca[r].size() && tier_ok;
                 ++l) {
                for (const auto& sh : src_v4_tiers.hca[r][l]) {
                    auto h = copy_page(sh, hca_bytes, memory::Pool::kHca);
                    if (!h) { tier_ok = false; break; }
                    dst_tiers.hca[r][l].push_back(*h);
                }
            }
            dst_tiers.lid[r].resize(src_v4_tiers.lid[r].size());
            for (size_t l = 0; l < src_v4_tiers.lid[r].size() && tier_ok;
                 ++l) {
                for (const auto& sh : src_v4_tiers.lid[r][l]) {
                    auto h = copy_page(sh, lid_bytes,
                                       memory::Pool::kIndexerK);
                    if (!h) { tier_ok = false; break; }
                    dst_tiers.lid[r][l].push_back(*h);
                }
            }
        }
        // Executor state rings + step tracking (throws on alloc failure).
        if (tier_ok && deps_.dcp_executor) {
            try {
                deps_.dcp_executor->v4_fork_sequence(src_id, dst_id);
            } catch (const std::exception& e) {
                spdlog::error("seq_fork: v4_fork_sequence failed: {}",
                              e.what());
                tier_ok = false;
            }
        }
        if (!tier_ok) {
            free_tier_pages();
            for (auto& hh : dst_ik) deps_.page_allocator->free(hh);
            for (auto& hh : dst_handles)
                if (hh.page_idx >= 0) deps_.page_allocator->free(hh);
            if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kSeqFork,
                        "seq_fork: V4 side-tier/state-ring clone failed "
                        "(pool or device alloc exhausted)");
            return;
        }
        // Cold-page copies (host bytes; no-op when tiering is off or the
        // parent has no demotions).
        if (v4_kv_tiering_) v4_kv_tiering_->on_seq_fork(src_id, dst_id);
    }

    // TD-PREFIX-FORK-COV (prefix-cache root cause, 2026-08-18): the DSA
    // indexer COVERAGE state machine (indexer_cov: next_pos high-water mark
    // + kUnset/kPaged/kArena/kDead mode) is per-seq and was not forked — the
    // child's first append at pos >= 1 read next_pos=0, was classified a GAP
    // and went PERMANENTLY kDead: every forked child silently decoded with
    // DENSE attention while a fresh sequence ran DSA sparse, flipping
    // near-tie tokens (different kernel accumulation over the same context)
    // and losing sparse perf for the child's whole life. Clone the parent's
    // coverage so the child's appends continue from the forked frontier.
    // kArena is executor-arena storage pinned to the PARENT's B==1 rows —
    // the child cannot append there; fail closed to dense (kDead).
    IndexerCov dst_cov = src_cov;
    if (dst_cov.mode == IndexerSeqMode::kArena)
        dst_cov.mode = IndexerSeqMode::kDead;

    const auto forked_count = static_cast<uint32_t>(dst_handles.size());
    // INV-SEQ-FORK-STATE enumeration point (fork-child construction): every
    // SequenceState member gets an explicit value or an explicit exclusion.
    {
        SequenceState st;
        auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers] = st;
        kv_pages = std::move(dst_handles);
        forked = true;
        indexer_pages = std::move(dst_ik);
        indexer_cov = dst_cov;
        // TD-V4-SERVE-PREFIX (2026-08-22): V4 side tiers are COPY-ON-FORK
        // (fresh pages + D2D above; the executor state rings were cloned by
        // v4_fork_sequence) — the tiers are mutate-in-place structures, so
        // sharing would corrupt the parent.
        v4_tiers = std::move(dst_tiers);
        sequences_.emplace(dst_id, std::move(st));
    }
    invalidate_kv_meta();  // invalidate dirty guard
    ++page_budget_.active_sequences;
    ++page_budget_.active_forks;

    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, dst_id, forked_count, 0);
}

// ── KD-4e1: Automatic KV page growth ─────────────────────────────────────

bool CommandDispatcher::ensure_pages(uint64_t seq_id, uint32_t token_pos) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return true;  // unknown seq — caller's check

    auto& pages = it->second.kv_pages;
    const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
    // TD-GOLDEN: pages are layer-major per logical page — counts below are
    // LOGICAL pages; each grows kv_layers_ physical pages at once.
    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    const int logical_have = static_cast<int>(pages.size()) / L;
    const int needed = static_cast<int>(token_pos / page_size) + 1;

    if (needed <= logical_have)
        return true;  // enough pages

    // Grow by at least chunk_size, at most to max_blocks_per_seq_.
    const int cap = max_blocks_per_seq_ > 0 ? max_blocks_per_seq_ : needed;
    const int target = std::min(
        std::max(needed, logical_have + chunk_size_pages_),
        cap);

    auto* pa = deps_.page_allocator;
    const auto& dcp = pa->dcp_config();
    const auto pool = pages.empty() ? memory::Pool::kMain : pages[0].pool;
    const int default_gpu = pages.empty() ? 0 : pages[0].gpu_idx;

    // INV-KV-REP: replicated KV growth claims each page on EVERY TP GPU in
    // lockstep (same canonical index) — owner routing here would recreate
    // the TD-KV-REPLICATED-PAGE-ALIAS aliasing on grown pages.  Applies to
    // BOTH kMain and kSpeculation (TD-KV-REPLICATED-SPEC): a draft
    // sequence's grown pages are replicated to every rank's block table
    // exactly like main pages.
    const bool replicated_kv = dcp.enabled() && !dcp.kv_sharded;

    for (int i = logical_have; i < target; ++i) {
        const auto ts = static_cast<uint32_t>(i) * page_size;

        // Sharded DCP routes pages to the owning GPU by token position.
        const int gpu = (!replicated_kv && dcp.enabled()
                         && pool == memory::Pool::kMain)
            ? pa->dcp_gpu_for_token(ts) : default_gpu;

        for (int l = 0; l < L; ++l) {
            // TD-V4-KMAIN-SIZING: V4 non-CSA layers grow sentinel slots
            // (side pools provision their KV in ensure_v4_tier_pages).
            if (!layer_takes_kmain_page(l)) {
                memory::PageHandle s;
                s.gpu_idx = gpu;
                s.page_idx = -1;
                s.gpu_ptr = nullptr;
                s.pool = pool;
                pages.push_back(s);
                continue;
            }
            // allocate_unreserved / unreserved=true: page growth is what
            // headroom protects (INV-4.9f).
            auto h = replicated_kv
                ? pa->allocate_replicated(seq_id, ts,
                                          static_cast<uint32_t>(l), pool,
                                          /*unreserved=*/true)
                : pa->allocate_unreserved(gpu, pool);
            if (!h) {
                spdlog::warn("ensure_pages: pool exhausted at logical page "
                             "{}/{} layer {} for seq {}", i, target, l, seq_id);
                // Roll back this logical page's partial layer set so the
                // layer-major indexing stays consistent (sentinels hold no
                // VRAM — skip the allocator free).
                while (static_cast<int>(pages.size()) % L != 0) {
                    if (pages.back().page_idx >= 0) pa->free(pages.back());
                    pages.pop_back();
                }
                // TD-GOLDEN-KV-EXHAUST: report failure when the REQUESTED
                // position is still uncovered (chunk over-growth being cut
                // short is not an error).
                return needed <= static_cast<int>(pages.size()) / L;
            }
            auto& m = pa->meta(*h);
            m.sequence_id = seq_id;
            m.token_start = ts;
            m.token_end = ts + static_cast<uint32_t>(page_size);
            m.layer_index = static_cast<uint32_t>(l);
            pages.push_back(*h);
        }
    }
    return true;
}

// ── Extended sequence + config handlers ────────────────────────────────────

void CommandDispatcher::handle_e_seq_create(const ipc::Command& cmd) {
    // E_CMD_SEQ_CREATE: extended sequence create. Currently delegates to the
    // atomic handler — the E-extension (KV init, optimized DCP routing) is an
    // internal C++ improvement that doesn't change the handler structure.
    handle_seq_create(cmd);
}

void CommandDispatcher::handle_e_seq_free(const ipc::Command& cmd) {
    // E_CMD_SEQ_FREE: extended sequence free. Currently delegates to the atomic
    // handler — the E-extension (NVMe spill decisions on teardown) is an
    // internal C++ optimization.
    handle_seq_free(cmd);
}

void CommandDispatcher::handle_config_update(const ipc::Command& cmd) {
    if (!deps_.live_config) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kConfigUpdate,
                    "config_update: no live config");
        return;
    }
    uint32_t applied = apply_config_update(*deps_.live_config, cmd);
    write_compute_completion(ipc::CMD_CONFIG_UPDATE, cmd.cmd_seq,
                             cmd.gpu_idx, applied, 0);
}


// ── CMD_SEQ_SNAPSHOT / CMD_SEQ_RESTORE (debug/test-only) ────────────────────
//
// Checkpoint a sequence's cached state so long-context integration tests can
// restart mid-prompt instead of re-forcing thousands of tokens. Path from env
// LS_SEQ_CKPT_PATH (the test runs in-process). File layout (native-endian):
//   header:  magic 'LSCK', version u32, token_count u32, kv_layers u32,
//            kv_page_size u32, kv_stride_block u64, logical_pages u32,
//            ik_page_tokens u32, ik_page_bytes u64, ik_groups u32,
//            dcp_size u32
//   body:    logical_pages*kv_layers blobs of kv_stride_block bytes
//            (logical-major, layer within — rank-0 content; replicated KV
//            means every rank's content is identical), then ik_groups blobs
//            of ik_page_bytes ((page, computing-layer)-ordered, rank 0).
// Restore requires a freshly created sequence with matching config; it grows
// pages via the SAME ensure paths the decode loop uses, uploads content to
// EVERY rank, and marks indexer coverage kPaged up to token_count.
// KVS-2: version 2 = the checkpoint was taken under SHARDED KV
// (hardware.dcp_kv_mode) — same layout, but each page blob is the OWNING
// rank's content (read from / restored to the handle's GPU only). Version
// encodes the mode; cross-mode restore is rejected.

namespace {
struct SeqCkptHeader {
    uint32_t magic = 0x4B43534CU;  // 'LSCK'
    uint32_t version = 1;
    uint32_t token_count = 0;
    uint32_t kv_layers = 0;
    uint32_t kv_page_size = 0;
    uint64_t kv_stride_block = 0;
    uint32_t logical_pages = 0;
    uint32_t ik_page_tokens = 0;
    uint64_t ik_page_bytes = 0;
    uint32_t ik_groups = 0;
    uint32_t dcp_size = 0;
};
}  // namespace

void CommandDispatcher::handle_seq_snapshot(const ipc::Command& cmd) {
    const uint64_t seq_id = cmd.seq_ckpt.seq_id;
    const uint32_t count = cmd.seq_ckpt.token_count;
    const char* path = std::getenv("LS_SEQ_CKPT_PATH");
    auto fail = [&](const char* msg) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqCreate,
                    msg);
    };
    if (!path || !*path) return fail("seq_snapshot: LS_SEQ_CKPT_PATH unset");
    if (!deps_.page_allocator || count == 0)
        return fail("seq_snapshot: bad state");
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return fail("seq_snapshot: unknown seq");
    const auto& seq_kv_pages = it->second.kv_pages;

    // GLM-25k (TD-KVT-SPEC resolved): a tiered sequence's demoted pages live
    // in the manager's pinned cold pools, not VRAM.  Drain in-flight
    // demotions so every page is either HOT (valid handle → VRAM read) or
    // COLD (host copy complete → read from the cold pool below); the
    // resulting checkpoint is byte-identical to a non-tiered snapshot of the
    // same state (INV-KVT-1 placement-only).
    const bool tiered_seq =
        kv_tiering_ && kv_tiering_->seq_has_demotions(seq_id);
    if (tiered_seq) kv_tiering_->drain_demotions();

    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
    const int64_t stride = deps_.kv_cache_stride_block;
    const int logical = (static_cast<int>(count) + page_size - 1) / page_size;
    if (static_cast<int>(seq_kv_pages.size()) < logical * L || stride <= 0
        || kv_cache_base_ptrs_.empty() || !kv_cache_base_ptrs_[0])
        return fail("seq_snapshot: kv state incomplete");

    const int dcp = deps_.dcp_executor ? deps_.dcp_executor->dcp_size() : 1;
    // KVS-2: sharded KV — page content lives ONLY on the owning rank's GPU;
    // snapshot reads each page from its handle's GPU (version 2 file). Under
    // replication every rank's content is identical → rank-0 reads, version 1
    // (byte-identical to the legacy format).
    const bool sharded = kv_sharded_ && dcp >= 2;
    const auto& pair0 = deps_.hidden_state_pairs[0];
    auto* dev0 = deps_.device_backends[pair0.gpu_position];
    dev0->set_device();

    SeqCkptHeader h{};
    h.version = sharded ? 2 : 1;
    h.token_count = count;
    h.kv_layers = static_cast<uint32_t>(L);
    h.kv_page_size = static_cast<uint32_t>(page_size);
    h.kv_stride_block = static_cast<uint64_t>(stride);
    h.logical_pages = static_cast<uint32_t>(logical);
    h.dcp_size = static_cast<uint32_t>(dcp);

    // Indexer pages: handles are (page, layer, rank)-ordered — dcp handles
    // per group under replicated indexer, ONE (the owner's) under local
    // (TD-GLM-INDEXER-LOCAL-MERGE). One representative per group suffices:
    // page CONTENT for global page g is mode-invariant (same FP8 rows +
    // scales at the same in-page offsets), only its placement differs, so a
    // checkpoint restores across indexer modes as long as ik_page_tokens
    // match. Record the owning GPU alongside each pointer — under local
    // mode the single copy lives on the owner rank's GPU, not rank 0's.
    const bool ik_local = dcp >= 2 && deps_.live_config
        && deps_.live_config->hardware.dcp_indexer_mode
               == config::DcpIndexerMode::local;
    const int ik_group_stride = ik_local ? 1 : dcp;
    const auto& seq_ik = it->second.indexer_pages;
    std::vector<std::pair<const void*, int>> ik_reps;  // (ptr, gpu position)
    int PT = 0;
    int64_t ik_bytes = 0;
    if (!seq_ik.empty() && deps_.live_config) {
        PT = deps_.live_config->memory.kv_cache.indexer_k_page_size_tokens;
        const int ihd = deps_.live_config->model.index_head_dim;
        ik_bytes = static_cast<int64_t>(PT) * ihd
                 + static_cast<int64_t>(PT) * sizeof(float);
        for (size_t i = 0; i < seq_ik.size(); i += ik_group_stride)
            ik_reps.emplace_back(seq_ik[i].gpu_ptr, seq_ik[i].gpu_idx);
    }
    h.ik_page_tokens = static_cast<uint32_t>(PT);
    h.ik_page_bytes = static_cast<uint64_t>(ik_bytes);
    h.ik_groups = static_cast<uint32_t>(ik_reps.size());

    std::FILE* f = std::fopen(path, "wb");
    if (!f) return fail("seq_snapshot: cannot open path");
    std::fwrite(&h, sizeof(h), 1, f);

    std::vector<std::byte> buf(static_cast<size_t>(stride));
    auto* base0 = static_cast<std::byte*>(kv_cache_base_ptrs_[0]);
    for (int j = 0; j < logical; ++j) {
        for (int l = 0; l < L; ++l) {
            const auto& ph = seq_kv_pages[static_cast<size_t>(j) * L + l];
            if (tiered_seq && ph.page_idx < 0) {
                // TD-KVT-SPEC: demoted page — the handle is neutralized and
                // the VRAM page freed; the exact bytes live in the tiering
                // manager's pinned cold pool (the round-robin cold owner's
                // single copy under replicated dedup, rank 0's replica
                // without it, the shard owner's copy under sharded —
                // content-identical every way, INV-KVT-1/-11).
                const void* cold =
                    kv_tiering_->cold_page_host_ptr(seq_id, l, j);
                if (!cold) {
                    std::fclose(f);
                    return fail("seq_snapshot: demoted page missing from "
                                "the cold pool (kv_tiering)");
                }
                std::fwrite(cold, 1, static_cast<size_t>(stride), f);
                continue;
            }
            if (ph.page_idx < 0 || !ph.gpu_ptr) {
                // TD-V4-KMAIN-SIZING: V4 non-CSA layer sentinel — no kMain
                // page exists (KV lives in the side tiers, which this
                // checkpoint format does not carry). Emit a zero block to
                // keep the fixed-stride file layout; restore skips it.
                std::vector<std::byte> zero(static_cast<size_t>(stride),
                                            std::byte{0});
                std::fwrite(zero.data(), 1, static_cast<size_t>(stride), f);
                continue;
            }
            if (sharded) {
                // Read from the OWNING rank's GPU pool (handle.gpu_idx is
                // the gpu position; content exists only there).
                auto* dev = deps_.device_backends[ph.gpu_idx];
                auto* base = static_cast<std::byte*>(
                    deps_.page_allocator->kv_main_base(ph.gpu_idx));
                if (!base) { std::fclose(f);
                             return fail("seq_snapshot: no pool base"); }
                dev->set_device();
                dev->memcpy_d2h_async(buf.data(),
                                      base + static_cast<int64_t>(ph.page_idx)
                                                 * stride,
                                      static_cast<size_t>(stride), nullptr);
                dev->synchronize_device();
            } else {
                dev0->memcpy_d2h_async(buf.data(),
                                       base0 + static_cast<int64_t>(ph.page_idx) * stride,
                                       static_cast<size_t>(stride), nullptr);
                dev0->synchronize_device();
            }
            std::fwrite(buf.data(), 1, static_cast<size_t>(stride), f);
        }
    }
    if (ik_bytes > 0) {
        std::vector<std::byte> ikbuf(static_cast<size_t>(ik_bytes));
        for (const auto& [p, gpu] : ik_reps) {
            // Read from the GPU that HOLDS the page (owner under local mode;
            // rank 0's replica under replicated).
            auto* dev = deps_.device_backends[gpu];
            dev->set_device();
            dev->memcpy_d2h_async(ikbuf.data(), p,
                                  static_cast<size_t>(ik_bytes), nullptr);
            dev->synchronize_device();
            std::fwrite(ikbuf.data(), 1, static_cast<size_t>(ik_bytes), f);
        }
    }
    std::fclose(f);
    spdlog::info("seq_snapshot: seq {} @{} tokens -> {} ({} kv pages, {} ik "
                 "groups)", seq_id, count, path, logical * L, h.ik_groups);
    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id,
                         static_cast<uint32_t>(logical * L), 0);
}

void CommandDispatcher::handle_seq_restore(const ipc::Command& cmd) {
    const uint64_t seq_id = cmd.seq_ckpt.seq_id;
    const char* path = std::getenv("LS_SEQ_CKPT_PATH");
    auto fail = [&](const char* msg) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqCreate,
                    msg);
    };
    if (!path || !*path) return fail("seq_restore: LS_SEQ_CKPT_PATH unset");
    if (!deps_.page_allocator) return fail("seq_restore: no allocator");
    auto seq_it = sequences_.find(seq_id);
    if (seq_it == sequences_.end())
        return fail("seq_restore: sequence must be created first");
    // GLM-25k (TD-KVT-SPEC-FORK lifted): restore writes pages via their VRAM
    // handles, and a demoted sequence's handles are freed/neutralized — so
    // restoring ONTO it first RE-PROMOTES every cold page (the exact demoted
    // bytes return to fresh VRAM pages through the standard growth path,
    // INV-KVT-1; seq_pages_ handles un-neutralized, kv-meta dirty guard
    // poisoned, cold slots released respecting fork-family refcounts).  The
    // body below then writes through valid handles into a clean, fully-hot
    // sequence — exactly the restore-into-fresh protocol — and tiering
    // re-demotes on subsequent decode steps.  Checkpoints WRITTEN from a
    // tiered sequence restore here unmodified (the snapshot captured both
    // tiers into the ordinary page-blob format).  VRAM-full keeps
    // fail-closed (capacity, not correctness).
    if (kv_tiering_ && kv_tiering_->seq_has_demotions(seq_id)
        && !kv_tiering_->repromote_seq(seq_id, /*keep_frontier=*/0))
        return fail("seq_restore: cold-page re-promotion failed (VRAM "
                    "capacity) — restore onto the demoted sequence fails "
                    "closed (TD-KVT-SPEC-FORK)");

    std::FILE* f = std::fopen(path, "rb");
    if (!f) return fail("seq_restore: cannot open path");
    SeqCkptHeader h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1 || h.magic != 0x4B43534CU
        || (h.version != 1 && h.version != 2)) {
        std::fclose(f);
        return fail("seq_restore: bad header");
    }
    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
    const int64_t stride = deps_.kv_cache_stride_block;
    const int dcp = deps_.dcp_executor ? deps_.dcp_executor->dcp_size() : 1;
    // KVS-2: version encodes the KV placement mode the checkpoint was taken
    // under (1 = replicated, 2 = sequence-sharded). Restoring across modes
    // would place content on the wrong rank(s) — reject.
    const bool sharded = kv_sharded_ && dcp >= 2;
    if (h.kv_layers != static_cast<uint32_t>(L)
        || h.kv_page_size != static_cast<uint32_t>(page_size)
        || h.kv_stride_block != static_cast<uint64_t>(stride)
        || h.dcp_size != static_cast<uint32_t>(dcp)
        || (h.version == 2) != sharded) {
        std::fclose(f);
        return fail("seq_restore: config mismatch vs checkpoint");
    }

    // Grow main KV to cover [0, token_count) via the standard growth path.
    if (!ensure_pages(seq_id, h.token_count - 1)) {
        std::fclose(f);
        return fail("seq_restore: kv page growth failed");
    }
    const auto& pages = seq_it->second.kv_pages;
    if (pages.size() < static_cast<size_t>(h.logical_pages) * L) {
        std::fclose(f);
        return fail("seq_restore: kv pages short after growth");
    }

    // Upload each page blob: replicated KV → EVERY rank (same page_idx
    // against each rank's pool base); sharded KV (KVS-2) → ONLY the owning
    // rank's GPU (the handle's gpu position — the same routing the page
    // allocation used, so content lands exactly where attention reads it).
    std::vector<std::byte> buf(static_cast<size_t>(stride));
    for (uint32_t j = 0; j < h.logical_pages; ++j) {
        for (int l = 0; l < L; ++l) {
            if (std::fread(buf.data(), 1, static_cast<size_t>(stride), f)
                != static_cast<size_t>(stride)) {
                std::fclose(f);
                return fail("seq_restore: truncated kv body");
            }
            const auto& ph = pages[static_cast<size_t>(j) * L + l];
            // TD-V4-KMAIN-SIZING: V4 non-CSA sentinel — the checkpoint block
            // is a zero placeholder (snapshot wrote it as such); nothing to
            // upload.
            if (ph.page_idx < 0 || !ph.gpu_ptr) continue;
            if (sharded) {
                auto* base = static_cast<std::byte*>(
                    deps_.page_allocator->kv_main_base(ph.gpu_idx));
                if (!base) { std::fclose(f);
                             return fail("seq_restore: no pool base"); }
                auto* dev = deps_.device_backends[ph.gpu_idx];
                dev->set_device();
                dev->memcpy_h2d(
                    base + static_cast<int64_t>(ph.page_idx) * stride,
                    buf.data(), static_cast<size_t>(stride));
            } else {
                for (int r = 0; r < dcp
                     && r < static_cast<int>(kv_cache_base_ptrs_.size()); ++r) {
                    const auto& pr = deps_.hidden_state_pairs[r];
                    auto* dev = deps_.device_backends[pr.gpu_position];
                    dev->set_device();
                    dev->memcpy_h2d(
                        static_cast<std::byte*>(kv_cache_base_ptrs_[r])
                            + static_cast<int64_t>(ph.page_idx) * stride,
                        buf.data(), static_cast<size_t>(stride));
                }
            }
        }
    }

    // Indexer-K: grow via the standard path, then upload (page, layer)
    // groups — to every rank's replica page under replicated indexer, or to
    // the single OWNER-rank page under local (TD-GLM-INDEXER-LOCAL-MERGE;
    // ensure_indexer_pages provisions the local shape, and group g's content
    // is mode-invariant so cross-indexer-mode restore is valid). OPTIONAL:
    // when provisioning is impossible (pool exhaustion) or the checkpoint's
    // indexer page size differs from the current config (content layout
    // mismatch), restore the KV only and mark coverage kDead — sparse stays
    // off for the sequence (dense is always correct), which is exactly the
    // dense-control semantics.
    const bool ik_local = dcp >= 2 && deps_.live_config
        && deps_.live_config->hardware.dcp_indexer_mode
               == config::DcpIndexerMode::local;
    const int ik_group_stride = ik_local ? 1 : dcp;
    const int PT_now = deps_.live_config
        ? deps_.live_config->memory.kv_cache.indexer_k_page_size_tokens : 0;
    if (h.ik_groups > 0
        && (h.ik_page_tokens != static_cast<uint32_t>(PT_now)
            || ensure_indexer_pages(seq_id, h.token_count - 1, 0, dcp)
                   != IndexerPageResult::kOk)) {
        seq_it->second.indexer_cov = {h.token_count, IndexerSeqMode::kDead};
        spdlog::warn("seq_restore: indexer pages unavailable or page-size "
                     "mismatch (ckpt {} vs config {}) — restored KV only; "
                     "sequence pinned to dense",
                     h.ik_page_tokens, PT_now);
    } else if (h.ik_groups > 0) {
        const auto& ik = seq_it->second.indexer_pages;
        if (ik.size() < static_cast<size_t>(h.ik_groups) * ik_group_stride) {
            std::fclose(f);
            return fail("seq_restore: indexer pages short after growth");
        }
        std::vector<std::byte> ikbuf(static_cast<size_t>(h.ik_page_bytes));
        for (uint32_t g = 0; g < h.ik_groups; ++g) {
            if (std::fread(ikbuf.data(), 1, ikbuf.size(), f) != ikbuf.size()) {
                std::fclose(f);
                return fail("seq_restore: truncated indexer body");
            }
            for (int r = 0; r < ik_group_stride; ++r) {
                const auto& ph =
                    ik[static_cast<size_t>(g) * ik_group_stride + r];
                // Upload to the GPU that HOLDS this handle's page (the owner
                // under local mode; each rank's replica under replicated —
                // handles are rank-ordered, so index r is rank r's copy).
                auto* dev = deps_.device_backends[ph.gpu_idx];
                dev->set_device();
                dev->memcpy_h2d(ph.gpu_ptr, ikbuf.data(), ikbuf.size());
            }
        }
        seq_it->second.indexer_cov = {h.token_count, IndexerSeqMode::kPaged};
    }
    std::fclose(f);

    invalidate_kv_meta();  // poison the dirty guard
    spdlog::info("seq_restore: seq {} <- {} @{} tokens ({} kv pages, {} ik "
                 "groups, dcp={})", seq_id, path, h.token_count,
                 h.logical_pages * L, h.ik_groups, dcp);
    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, h.token_count, 0);
}

}  // namespace layerstorm::daemon
