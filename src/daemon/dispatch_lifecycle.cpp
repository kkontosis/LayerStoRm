// Cache, graph, stream, placement, sequence, and config handlers.
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/v4_kv_tiering.h"

#include <algorithm>  // std::fill / std::min (R4a truncating fork)
#include <cstdio>
#include <cstdlib>
#include <cstring>   // GF3.12: checkpoint tail masking (memcpy/memset)
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
#include "model/model_config.h"  // TD-MAXSEQ precheck: computes_indexer
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

bool CommandDispatcher::claim_kda_state(uint64_t seq_id, int default_gpu,
                                        std::vector<memory::PageHandle>& out,
                                        std::string& err) {
    out.clear();
    auto* pa = deps_.page_allocator;
    // No KDA pool on this model (every non-glm5_next arch): no-op, empty
    // handles — behavior byte-identical to pre-GF3.8.
    if (!pa || pa->kda_state_slot_bytes() <= 0) return true;
    // Engaged GPUs: the TP set when configured (each rank holds its H/tp
    // head shard, GF3.10), else the create-command GPU. Per-rank slot
    // indices are INDEPENDENT (nothing consumes cross-rank index equality;
    // the S4 lockstep finding) but the claim is all-or-nothing.
    std::vector<int> gpus;
    const auto& dcp = pa->dcp_config();
    if (!dcp.tp_gpu_indices.empty()) gpus = dcp.tp_gpu_indices;
    else gpus.push_back(default_gpu);
    // TD-KDA-STATE-MAPPED-SLABS: one allocator call per GPU claims the
    // sequence's COMPLETE state there — carve mode: 1 whole-request slot
    // handle; mapped mode: num_layers per-layer unit handles (contiguous
    // whole-slab runs over the shared region). Handles are appended
    // RANK-MAJOR: out = [gpu0 unit0..unitN, gpu1 unit0..unitN, ...] —
    // every downstream consumer indexes [rank * units + unit].
    const auto unit_bytes = static_cast<size_t>(pa->kda_unit_bytes());
    for (int g : gpus) {
        auto hs = pa->allocate_kda_state(g, seq_id);
        if (hs.empty()) {
            for (auto& hh : out) pa->free(hh);
            out.clear();
            // "exhausted" leads the message: the orchestrator classifies by
            // CATEGORY, and the 80-byte field truncates (INV-IPC-ERRMSG-80).
            // Mapped mode carries the TD-KDA-MAPPED-FRAG discriminator:
            // FRAG = contiguity shortage (free slabs cover the demand but
            // too few contiguous runs exist — eviction returns bytes, not
            // adjacency), capacity = byte shortage (eviction helps).
            if (pa->kda_state_mapped()) {
                const auto& st = pa->kda_mapped_stats(g);
                err = "exhausted KDA state gpu " + std::to_string(g)
                    + (st.last_was_contiguity
                           ? ": FRAG " + std::to_string(st.last_found_runs)
                                 + "/" + std::to_string(st.last_needed_runs)
                                 + " runs, "
                                 + std::to_string(st.last_free_slabs)
                                 + " slabs free"
                           : ": capacity "
                                 + std::to_string(st.last_free_slabs)
                                 + " slabs free, need "
                                 + std::to_string(
                                       pa->kda_units_per_rank()
                                       * pa->kda_unit_slabs()));
            } else {
                err = "exhausted KDA state pool on gpu " + std::to_string(g)
                    + " (" + std::to_string(
                          pa->free_pages(g, memory::Pool::kKdaState))
                    + "/" + std::to_string(
                          pa->total_pages(g, memory::Pool::kKdaState))
                    + " slots free)";
            }
            return false;
        }
        // INV-V4-DET obligation (2), verbatim: ZERO ON CLAIM — a reused
        // slot otherwise hands this sequence the previous holder's
        // recurrent state + conv rings (mapped mode: a reused SLAB
        // otherwise hands it stale KV/indexer bytes). The memsets ride the
        // per-GPU kAttention stream, ordering them before every later
        // state read/write of this sequence (GF3.9 must keep KDA state
        // kernels on that stream — the same ordering contract the V4 tier
        // zeroing and the fork D2D rely on). Only the LOGICAL unit bytes
        // are zeroed; a mapped run's tail padding is never read by any
        // kernel, copy or checkpoint.
        if (static_cast<size_t>(g) < deps_.device_backends.size()
            && deps_.device_backends[static_cast<size_t>(g)]
            && deps_.stream_manager) {
            auto* be = deps_.device_backends[static_cast<size_t>(g)];
            be->set_device();
            for (auto& h : hs)
                be->memset_async(h.gpu_ptr, 0, unit_bytes,
                                 deps_.stream_manager->stream(
                                     g, compute::StreamId::kAttention));
        }
        out.insert(out.end(), hs.begin(), hs.end());
    }
    return true;
}

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

    // ── TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA: self-inflicted-capacity
    // fast-fail. A seq_create claims THREE tenants from the ONE shared
    // kMain pool: the KV pages just computed, the mapped KDA state's
    // whole-slab runs, and the full-length indexer-K reservation
    // (INV-DSA-RESERVE, below). When the request's OWN whole-life demand
    // exceeds the pool's TOTAL capacity, the retryable evict-a-holder
    // remedy cannot possibly help — no eviction frees the request's own
    // footprint (the GF3 1M-arm incident: KV 10,816 slabs + state 578 vs
    // an 11,319-slab pool spilled two holders, retried five times, and
    // 500'd). Refuse FAST and NON-RETRYABLE, quoting the boot-computed
    // admissible ceiling. Holder-inflicted shortages (own demand <= pool)
    // keep the retryable kKvPoolExhausted path untouched (INV-KDA-STATE,
    // TD-KDA-MAPPED-RETRY-EFFECTIVE). Windowed admission already shrank
    // num_pages above, so tiered shapes are exempt by construction; the
    // sharded-KV / local-indexer per-rank division is skipped
    // conservatively (per-rank demand is a fraction of the whole).
    if (pool == memory::Pool::kMain && deps_.live_config) {
        const bool kv_sharded_dcp = dcp.enabled() && dcp.kv_sharded;
        const bool idx_local = dcp.enabled() && dcp.indexer_k_sharded;
        if (!kv_sharded_dcp && !idx_local) {
            int precheck_layers = 0;
            const int all_layers = kv_layers_ > 0 ? kv_layers_ : 1;
            for (int l = 0; l < all_layers; ++l)
                if (layer_takes_kmain_page(l)) ++precheck_layers;
            const int64_t kv_need =
                static_cast<int64_t>(num_pages) * precheck_layers;
            const int pps = pa->pages_per_slab();
            int64_t state_need = 0;
            if (pa->kda_state_mapped() && pps > 0)
                state_need = static_cast<int64_t>(pa->kda_units_per_rank())
                             * pa->kda_unit_slabs() * pps;
            int64_t idx_need = 0;
            const int idx_pt = deps_.live_config->memory.kv_cache
                                   .indexer_k_page_size_tokens;
            if (p.reserve_tokens > 0 && model_has_paged_indexer()
                && pps > 0 && idx_pt > 0) {
                const auto win = static_cast<uint32_t>(
                    deps_.live_config->serving.max_sequence_length);
                const uint32_t rt = std::min(p.reserve_tokens, win);
                model::ModelConfig mc(*deps_.live_config);
                const int n_lay =
                    deps_.live_config->model.num_hidden_layers
                    + deps_.live_config->model.num_nextn_predict_layers;
                int computing = 0;
                for (int l = 0; l < n_lay; ++l)
                    if (mc.computes_indexer(l)) ++computing;
                idx_need = ((static_cast<int64_t>(rt) + idx_pt - 1) / idx_pt)
                           * computing * pps;
            }
            const int64_t total_need = kv_need + state_need + idx_need;
            const auto cap = static_cast<int64_t>(
                pa->total_pages(static_cast<int>(cmd.gpu_idx),
                                memory::Pool::kMain));
            if (cap > 0 && total_need > cap) {
                const int ceiling = pa->admissible_ctx_tokens();
                spdlog::warn(
                    "seq_create: seq {} REFUSED at admission — request "
                    "over kMain pool capacity: needs {} pages (KV {} + "
                    "KDA state {} + indexer {}) but gpu {} pool holds {} "
                    "TOTAL. Self-inflicted shortage: holder eviction "
                    "cannot help, refusing NON-RETRYABLE (admissible "
                    "single-request ceiling {} tokens; prompt_len {}; "
                    "TD-MAXSEQ-NOT-ADMISSIBLE-MAPPED-KDA)",
                    seq_id, total_need, kv_need, state_need, idx_need,
                    cmd.gpu_idx, cap,
                    ceiling > 0 ? std::to_string(ceiling) : "unknown",
                    prompt_len);
                // NOTE: the CMP message must NOT contain "exhausted" —
                // is_pool_exhaustion() substring-matches it and would arm
                // the evict-retry seam this refusal exists to bypass
                // (INV-IPC-ERRMSG-80: 80-byte field, ceiling front-loaded).
                std::string msg = "seq_create: over kMain pool capacity";
                if (ceiling > 0)
                    msg += " — admissible ~" + std::to_string(ceiling)
                           + " tokens";
                msg += " (need " + std::to_string(total_need) + "/"
                       + std::to_string(cap) + " pages)";
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kSeqCreate, msg.c_str());
                return;
            }
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
                // S2 (TD-INDEXER-POOL-ELASTIC): run-aware claim — on
                // slabbed models the page comes from this sequence's
                // position-major bump run (rule 3: one sequence per slab;
                // the position-outer/layer-inner loop above IS what makes
                // the packing position-major).
                h = pa->allocate_for_sequence(
                    static_cast<int>(cmd.gpu_idx), pool, seq_id,
                    static_cast<uint32_t>(l), token_start, token_end);
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
                // TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION (P-30 step 3): a
                // kMain bulk-KV shortage on a WITHIN-CAPACITY request (the
                // over-total-capacity class was refused non-retryably above)
                // is the retryable pool class — the same seam the KDA-state
                // and indexer sites already ride. It carried kSeqCreate with
                // no rebalancer poke, so a 97k prefill's refusals never
                // armed the eager drain and the orchestrator's bounded wait
                // waited on a drain nothing had started. Category-first
                // classification also survives INV-IPC-ERRMSG-80 truncation.
                // The shortfall (in slabs) makes the eager drain
                // demand-aware; kSpeculation stays kSeqCreate (the draft
                // pool is not the rebalancer's region).
                if (pool == memory::Pool::kMain) {
                    const int64_t pps =
                        std::max<int64_t>(1, pa->pages_per_slab());
                    const int64_t need_pages =
                        static_cast<int64_t>(num_pages) * real_layers;
                    pool_refusal_shortfall_slabs_ =
                        (std::max<int64_t>(0, need_pages - pool_free) + pps
                         - 1) / pps;
                }
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            pool == memory::Pool::kMain
                                ? ipc::CmpErrorCategory::kKvPoolExhausted
                                : ipc::CmpErrorCategory::kSeqCreate,
                            msg.c_str());
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

    // ── GF3.8: KDA per-request state slot — claimed at ADMISSION ──
    // The state does NOT grow with context, so this single claim covers the
    // sequence's whole life: the INV-DSA-RESERVE admission discipline holds
    // TRIVIALLY (there is no later provisioning step that could fail —
    // capacity is a slot count, checked exactly once, here). Failure is a
    // RETRYABLE kKvPoolExhausted refusal the orchestrator answers with
    // holder eviction + re-issue — never a mid-request failure. Draft
    // sequences (kSpeculation) claim nothing: the MTP draft is sparse MLA
    // and carries no recurrent state (MODELINFO section 5).
    std::vector<memory::PageHandle> kda_handles;
    std::array<std::vector<memory::PageHandle>, 2> kda_anchor_handles;
    if (pool == memory::Pool::kMain) {
        std::string kda_err;
        bool kda_ok = claim_kda_state(seq_id, static_cast<int>(cmd.gpu_idx),
                                      kda_handles, kda_err);
        // P-29 step 13 phase B: with MTP speculation armed, the TWO KDA anchor
        // slots (INV-KDA-REWIND anchor-and-replay) are claimed AT
        // ADMISSION like the live slot — the mid-round lazy claim turned
        // pool pressure into a mid-request CMP error; here it is the same
        // RETRYABLE holder-eviction seam every state claim rides.
        if (kda_ok && !kda_handles.empty() && deps_.live_config
            && deps_.live_config->speculation.enabled
            && deps_.live_config->speculation.method
                   == config::SpeculationMethodType::mtp
            && deps_.live_config->speculation.mtp.enabled) {
            for (int k = 0; k < 2 && kda_ok; ++k) {
                kda_ok = claim_kda_state(seq_id,
                                         static_cast<int>(cmd.gpu_idx),
                                         kda_anchor_handles[k], kda_err);
            }
            if (!kda_ok) {
                for (auto& av : kda_anchor_handles)
                    for (auto& hh : av) pa->free(hh);
                for (auto& hh : kda_handles) pa->free(hh);
                kda_err += " (MTP anchor slot)";
            }
        }
        if (!kda_ok) {
            for (auto& prev : handles)
                if (prev.page_idx >= 0) pa->free(prev);
            spdlog::warn(
                "seq_create: seq {} REFUSED at admission — KDA state slot "
                "claim failed: {} (retryable)",
                seq_id, kda_err);
            const std::string msg = "seq_create: " + kda_err
                + " — retryable, evict a prefix holder";
            // 44z: the rebalancer's policy-failure poke rides write_error's
            // kKvPoolExhausted choke point (INV-KVXP (b)).
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kKvPoolExhausted, msg.c_str());
            return;
        }
    }

    const auto allocated = static_cast<uint32_t>(handles.size());
    // INV-SEQ-FORK-STATE enumeration point (create): the structured binding
    // is the compile-time completeness gate — adding a SequenceState member
    // breaks it, forcing an explicit init decision here (and clone-vs-
    // exclude at fork, release at free).
    {
        SequenceState st;
        auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers,
               indexer_reserved_tokens, kda_state, kda_next_pos,
               kda_spill, kda_ckpts, mtp_indexer_cov, rewind_run,
               kda_anchors] = st;
        (void)kda_spill;  // GF3.12: a fresh sequence is never spilled —
                          // spill exists only for hibernated holders
        (void)kda_ckpts;  // P-29 step 24: captured later by D_CMD_KDA_CKPT
                          // at prefill cadence — a fresh sequence has none
        (void)mtp_indexer_cov;  // P-29 step 13: untracked until the first MTP
                                // append (fresh IndexerCov default)
        (void)rewind_run;       // P-29 step 13: no blessed window yet
        // P-29 step 13: anchor slots claimed at admission when MTP speculation
        // is armed (empty otherwise; the copy helpers keep a lazy-claim
        // fallback for probe/test producers).
        kda_anchors.slots[0] = std::move(kda_anchor_handles[0]);
        kda_anchors.slots[1] = std::move(kda_anchor_handles[1]);
        kv_pages = std::move(handles);
        forked = false;         // fresh create
        kda_state = std::move(kda_handles);  // GF3.8: claimed + zeroed above
        (void)indexer_pages;    // reserved below or grown lazily
        // Coverage stays untracked until the first attention step, but the
        // incarnation draws a fresh rewind epoch NOW (INV-DSA-EPOCH,
        // TD-INDEXER-STEPKEY-TOKEN-BLIND): a recycled seq_id must never
        // reproduce a dead incarnation's IndexShare step fingerprints.
        indexer_cov.epoch = ++indexer_epoch_next_;
        (void)v4_tiers;         // provisioned lazily (ensure_v4_tier_pages)
        indexer_reserved_tokens = 0;  // granted below on success
        (void)kda_next_pos;     // GF3.9: sized lazily at the first KDA
                                // stage; state was zeroed on claim, so the
                                // frontier starts at 0 for every layer
        sequences_.emplace(seq_id, std::move(st));
    }

    // TD-INDEXER-NO-DENSE-FALLBACK (Route 1): RESERVE AT ADMISSION.  Commit
    // the sequence's indexer-K pages NOW for the context it may ever reach
    // (reserve_tokens = prompt + generation budget + speculative-overshoot
    // margin, clamped to the serving window), so ensure_indexer_pages can
    // never fail later — the mid-prefill exhaustion that used to downgrade
    // an ADMITTED sequence to permanent dense (IndexerSeqMode::kDead) is
    // moved here, where failure is a clean RETRYABLE refusal the
    // orchestrator already answers with evict_for_admission + re-issue.
    // Deliberately NOT a fresh over-carve: the pool stays sized by
    // INV-KVT-14b (max_concurrent_requests x max_sequence_length,
    // VRAM-clamped); this claims only the ADMITTED request's own pages —
    // the same pages the sequence would have claimed lazily by end of life,
    // committed earlier so the failure surfaces at the admission seam.
    // Indexer-K is NOT tierable (top-k scoring touches every position each
    // step), so unlike windowed kMain admission there is no demote-behind
    // alternative: the full-length claim is the honest capacity cost of
    // serving this sequence SPARSELY.
    uint32_t granted_reserve = 0;
    if (pool == memory::Pool::kMain && p.reserve_tokens > 0
        && model_has_paged_indexer()) {
        const auto max_seq = static_cast<uint32_t>(
            deps_.live_config->serving.max_sequence_length);
        granted_reserve = std::min(p.reserve_tokens, max_seq);
        const int dcp = deps_.dcp_executor ? deps_.dcp_executor->dcp_size()
                                           : 1;
        const auto r = grow_indexer_pages(seq_id, granted_reserve - 1, dcp);
        if (r != IndexerPageResult::kOk) {
            auto itn = sequences_.find(seq_id);
            if (itn != sequences_.end()) {
                for (auto& h : itn->second.indexer_pages)
                    pa->free(h);
                for (auto& h : itn->second.kv_pages)
                    if (h.page_idx >= 0) pa->free(h);
                for (auto& h : itn->second.kda_state)  // GF3.8 rollback
                    pa->free(h);
                sequences_.erase(itn);
            }
            invalidate_kv_meta();
            const bool exhausted = r == IndexerPageResult::kExhausted;
            spdlog::warn(
                "seq_create: seq {} REFUSED at admission — indexer-K "
                "reservation for {} tokens {} (prompt_len {}; retryable={})",
                seq_id, granted_reserve,
                exhausted ? "exhausted the pool" : "is unavailable",
                prompt_len, exhausted);
            // "exhausted" early in the message + the CATEGORY: the
            // orchestrator classifies by category (INV-IPC-ERRMSG-80 — the
            // 80-byte field truncates), evicts a prefix holder, re-issues.
            // 44z: only the EXHAUSTED (retryable) class reaches the
            // rebalancer — via write_error's kKvPoolExhausted choke point;
            // an out-of-window reservation is not a capacity event and no
            // reclaim could have helped it.
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        exhausted ? ipc::CmpErrorCategory::kKvPoolExhausted
                                  : ipc::CmpErrorCategory::kSeqCreate,
                        exhausted
                            ? "seq_create: exhausted indexer-K pool at "
                              "admission reservation — retryable, evict a "
                              "prefix holder"
                            : "seq_create: indexer-K reservation unavailable "
                              "(beyond serving window or allocator unwired)");
            return;
        }
        sequences_[seq_id].indexer_reserved_tokens = granted_reserve;
    }
    invalidate_kv_meta();  // invalidate dirty guard
    ++page_budget_.active_sequences;

    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, allocated, 0,
                         granted_reserve);
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
    auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers,
           indexer_reserved_tokens, kda_state, kda_next_pos,
           kda_spill, kda_ckpts, mtp_indexer_cov, rewind_run,
           kda_anchors] = it->second;
    (void)indexer_cov;  // no external resources — dies with the aggregate
    (void)indexer_reserved_tokens;  // bookkeeping only — pages freed below
    (void)kda_next_pos;  // GF3.9: bookkeeping only — dies with the aggregate
    (void)mtp_indexer_cov;  // P-29 step 13: bookkeeping only
    (void)rewind_run;       // P-29 step 13: bookkeeping only
    // P-29 step 13: return the KDA anchor slots (claimed from Pool::kKdaState).
    for (auto& av : kda_anchors.slots)
        for (auto& h : av)
            if (h.page_idx >= 0) deps_.page_allocator->free(h);

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
    // GF3.8: return the KDA state slots. The slot is NOT recomputable
    // without full replay (INV-KDA-REWIND), so there is no tiering/demote
    // path for it — lifetime is exactly the sequence's lifetime, and
    // capacity pressure is answered at ADMISSION (holder eviction frees
    // slots through this very path), never by evicting a live sequence's
    // state.
    for (auto& h : kda_state) deps_.page_allocator->free(h);
    // GF3.12: release a spilled holder's host-resident state bytes (the
    // NumaManager buffers; heap fallbacks die with the aggregate).
    for (auto& r : kda_spill)
        if (r.buf.data && deps_.numa_manager) deps_.numa_manager->free(r.buf);
    // P-29 step 24: release the sequence's host KDA prefix checkpoints
    // (same storage discipline as kda_spill) and settle the global byte
    // counter — checkpoints belong to their prefix entry and die with its
    // holder sequence (or with a live request that never registered).
    if (!kda_ckpts.empty()) {
        const auto slot_b = static_cast<size_t>(
            std::max<int64_t>(0, deps_.page_allocator->kda_state_slot_bytes()));
        for (auto& [pos, ranks] : kda_ckpts) {
            (void)pos;
            const size_t bytes = slot_b * ranks.size();
            kda_ckpt_host_bytes_ -= std::min(kda_ckpt_host_bytes_, bytes);
            for (auto& r : ranks)
                if (r.buf.data && deps_.numa_manager)
                    deps_.numa_manager->free(r.buf);
        }
    }
    const bool was_fork = forked;
    sequences_.erase(it);
    invalidate_kv_meta();  // invalidate dirty guard
    if (page_budget_.active_sequences > 0) --page_budget_.active_sequences;
    if (was_fork && page_budget_.active_forks > 0) --page_budget_.active_forks;

    // S2 (TD-INDEXER-POOL-ELASTIC) fragmentation telemetry: per-GPU slab
    // occupancy after each teardown — fragmented = free pages stranded
    // inside claimed slabs (the tail-fragmentation cost the design gates
    // on; grep "S2 frag" in serve logs for the measurement).
    if (deps_.page_allocator->pages_per_slab() > 0) {
        const auto* pa = deps_.page_allocator;
        const double page_mib =
            static_cast<double>(pa->slab_bytes())
            / static_cast<double>(pa->pages_per_slab()) / (1024.0 * 1024.0);
        for (int g = 0; g < pa->gpu_count(); ++g) {
            const auto f = pa->kv_fragmentation(g);
            if (f.total_slabs == 0) continue;
            if (f.live_slabs == 0 && f.loose_free_pages == 0) continue;
            spdlog::info(
                "S2 frag gpu{}: {} live / {} free of {} slabs, {} used pages, "
                "{} fragmented free pages ({:.1f} MiB), {} loose",
                g, f.live_slabs, f.free_slabs, f.total_slabs, f.used_pages,
                f.fragmented_free_pages,
                f.fragmented_free_pages * page_mib, f.loose_free_pages);
        }
    }

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
    // R4a (RADIX_SLAB_DESIGN §5 R4): TRUNCATING fork.  prefix_len == 0 is
    // the legacy FULL fork and must stay byte-identical (every pre-R4
    // producer writes a zeroed slot); prefix_len > 0 forks only the
    // parent's first prefix_len tokens.  See the ipc_protocol.h payload
    // comment for the caller contract (committed length / rewind depth —
    // the engine tracks pages, not token counts, so it cannot verify).
    const uint32_t prefix_len = cmd.seq_fork.prefix_len;
    const bool truncated = prefix_len > 0;
    // R3 (TD-PREFIX-POOL-PRESSURE-EVICTS-THE-PRIZE): a FROZEN fork creates
    // a prefix HOLDER — a dst that never appends.  Both CoW frontier splits
    // are skipped (pure refcount share, zero page/slab cost in BOTH pools);
    // the next fork FROM the holder performs the CoW, and the parent's
    // continued appends touch only rows >= the fork point, which lie beyond
    // the holder's coverage (rewind never goes below the fork point either,
    // so no shared row the holder covers is ever rewritten).  V4 side tiers
    // are EXEMPT: they mutate in place at arbitrary slots (SWA ring,
    // compressor rings), so a frozen holder still takes the copy-on-fork
    // side-tier set (INV-PREFIX-CACHE-3 unchanged on V4).
    const bool frozen = cmd.cmd_type == ipc::CMD_SEQ_FORK_FROZEN;

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

    // TD-PREFIX-TIDY-COLD-SPILL: a hit on a SPILLED holder reloads its
    // cold pages from disk BEFORE the fork (the child shares refcounted
    // cold slots exactly as from a never-spilled holder).  Cold-slot
    // exhaustion is RETRYABLE — the orchestrator's holder-eviction seam
    // frees slots and re-issues the fork.  An unreadable file is the same
    // category (the orchestrator evicts THIS holder and treats the lookup
    // as a miss — capacity, never a failed request).
    if (kv_tiering_ && kv_tiering_->seq_spilled(src_id)
        && !kv_tiering_->unspill_seq(src_id)) {
        // 44z policy-failure poke: write_error's kKvPoolExhausted choke point.
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kKvPoolExhausted,
                    "seq_fork: spilled holder reload failed (cold pool "
                    "exhausted or spill file unreadable) — evict and retry "
                    "(TD-PREFIX-TIDY-COLD-SPILL)");
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
                 src_v4_tiers, src_indexer_reserved,
                 src_kda_state, src_kda_next_pos,
                 src_kda_spill, src_kda_ckpts, src_mtp_cov, src_rewind_run,
                 src_kda_anchors] = src_it->second;
    (void)src_forked;  // dst is a fork regardless of the parent's origin
    (void)src_indexer_reserved;  // child's reservation comes from the cmd
    (void)src_kda_anchors;   // P-29 step 13: round-transient — a child re-anchors
                             // (anchors exist only mid-speculation; forks
                             // happen between requests)
    // src_kda_next_pos is CLONED into the child below: the slot D2D copies
    // the recurrent state AT the parent's per-layer frontier (GF3.9).

    // R4a/R4b per-architecture applicability, gated on the ARCHITECTURE
    // PROPERTY (AttentionArch::lossy_position_indexed_state), not the
    // model name: a truncating fork is IMPOSSIBLE — and REJECTED, never
    // approximated — on any arch whose per-sequence state includes
    // in-place pos%capacity rings (V4: SWA ring + compressor state
    // rings).  The ring contents a prefix_len-truncated child needs
    // (positions [prefix_len - window, prefix_len)) were destroyed by the
    // parent's later appends unless prefix_len equals the parent frontier
    // (= a full fork); seeding from the CURRENT rings would silently
    // attend over post-boundary content.  A future ring-carrying arch is
    // excluded automatically; a ring-free arch gets truncation for free.
    // Belt-and-braces: a sequence that PHYSICALLY carries side-tier ring
    // state (src_v4_tiers) is rejected regardless of what the live config
    // claims (unit dispatchers flip configs mid-test).  The orchestrator
    // mirrors this gate via EngineInfo::seq_fork_truncatable and must not
    // offer mid-edge reuse where it is false.
    // GF3.8 belt-and-braces (same rule as src_v4_tiers): a sequence
    // PHYSICALLY carrying a KDA state slot is rejected for truncation
    // regardless of what the live config claims — the recurrent state has
    // no position axis at all, so state@prefix_len is algebraically
    // unrecoverable (INV-KDA-REWIND); reuse points exist only where a
    // CHECKPOINT was taken (GF3.12).
    //
    // P-29 step 24 (GF3.12 REALIZED, LS_KDA_PREFIX_CKPT): that checkpoint
    // now exists — a truncating fork on a KDA sequence is ADMITTED iff
    // the SOURCE holds a host checkpoint at EXACTLY prefix_len (64-aligned
    // by construction of the capture; asserted anyway). The child then
    // claims fresh state slots and H2D-restores the position-prefix_len
    // blob below (the bit-exact whole-slot round-trip, INV-KDA-CARRY) and
    // its per-layer frontier is set to prefix_len. Everything else about
    // the truncating fork (KV page share, indexer-K share, coverage
    // clamp, fresh epoch) is the EXISTING R4a machinery, unchanged. No
    // checkpoint at prefix_len ⇒ the refusal below stands — never
    // approximate (INV-KDA-ANCHOR discipline). V4 ring state has no
    // checkpoint story and stays refused unconditionally.
    const auto src_ckpt_it = src_kda_ckpts.find(prefix_len);
    const bool kda_ckpt_fork = truncated
        && src_ckpt_it != src_kda_ckpts.end()
        && !src_ckpt_it->second.empty()
        && src_v4_tiers.swa.empty()
        && prefix_len % 64 == 0;
    if (truncated && !kda_ckpt_fork
        && (!seq_fork_truncatable() || !src_v4_tiers.swa.empty()
            || !src_kda_state.empty() || !src_kda_spill.empty())) {
        // INV-IPC-ERRMSG-80: message must fit the 80-byte completion slot.
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kSeqFork,
                    "seq_fork: prefix_len unsupported: no KDA checkpoint "
                    "at prefix_len (lossy arch)");
        return;
    }

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
    if (kv_tiering_) kv_tiering_->on_seq_fork(src_id, dst_id, prefix_len);

    // R4a slab/page accounting decision (INV-SLAB-2 consistency): a
    // truncated child REFCOUNT-SHARES the parent's logical pages
    // [0, ceil(prefix_len / page_size)) exactly as a full fork shares all
    // of them — sharing stays BELOW the slab layer, every shared page
    // remains accounted to the ALLOCATING sequence's slab, and when the
    // parent later frees while the child holds only a partial range, the
    // parent's TAIL slabs (token ranges >= the boundary — position-major
    // packing makes those a contiguous slab suffix) return whole
    // immediately, while the shared-prefix slabs stay pinned by the
    // child's refs and return whole at the last fork-family ref drop (the
    // dead owner's run entry persists for accounting, receives no new
    // allocations).  The straddling logical page is NOT copied for the
    // share itself: a frozen holder never writes it, and a live child
    // gets the standard frontier CoW below before its first append.
    const size_t kv_l = static_cast<size_t>(kv_layers_ > 0 ? kv_layers_ : 1);
    // Same page-size authority as handle_seq_create's non-V4 arm (V4 is
    // rejected above): the DcpConfig value, which the engine sets to
    // memory.kv_cache.page_size_tokens.
    const uint32_t kv_ps = static_cast<uint32_t>(
        deps_.page_allocator->dcp_config().page_size_tokens);
    const size_t src_logical = src_kv_pages.size() / kv_l;
    // ceil(prefix_len / page_size), clamped to what the parent holds
    // (windowed admission can under-allocate vs token length; the child
    // grows the rest lazily through ensure_pages like anyone else).
    const size_t want_logical = truncated
        ? (static_cast<size_t>(prefix_len) + kv_ps - 1) / kv_ps
        : src_logical;
    const size_t take_logical = std::min(want_logical, src_logical);

    const auto& src_handles = src_kv_pages;
    std::vector<memory::PageHandle> dst_handles;
    dst_handles.reserve(truncated ? take_logical * kv_l
                                  : src_handles.size());

    const size_t take_handles =
        truncated ? take_logical * kv_l : src_handles.size();
    for (size_t i = 0; i < take_handles; ++i) {
        const auto& h = src_handles[i];
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
    // R4a: for a truncated LIVE child the split targets the STRADDLING
    // logical group — the page containing position prefix_len — and only
    // when the child will actually write it: prefix_len mid-page
    // (prefix_len % page_size != 0) AND the parent holds that page
    // (take_logical == want_logical).  A page-ALIGNED truncation needs no
    // split at all (the child's first append allocates a fresh logical
    // page through ensure_pages; every shared page is read-only for both
    // lifetimes), and a clamped truncation (parent under-allocated) never
    // touches a shared page either.  A COLD straddling group (hibernated
    // holder source, page_idx == -1 skipped here) is handled after the
    // child exists via repromote_for_rewind — the cold-slot CoW split.
    const size_t L = static_cast<size_t>(kv_layers_ > 0 ? kv_layers_ : 1);
    const bool kv_cow = !frozen && dst_handles.size() >= L
        && (!truncated
            || (prefix_len % kv_ps != 0 && take_logical == want_logical));
    if (kv_cow) {
        const size_t first = dst_handles.size() - L;
        for (size_t i = first; i < dst_handles.size(); ++i) {
            auto& h = dst_handles[i];
            // GLM-25k: the append-frontier (last logical) page never demotes
            // (INV-KVT-4), so these handles are always valid; guard anyway —
            // cow_copy on a neutralized handle would corrupt the allocator.
            if (h.page_idx < 0) continue;
            try {
                // S2: dst_id routes the split's slab placement (source-slab
                // colocation first, else the child's run) — holders stay
                // refcount-cheap on GLM (INV-PREFIX-CACHE-3).
                h = deps_.page_allocator->cow_copy(h, dst_id);
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
        // R4a: indexer-K truncation mirrors the KV slice on the indexer
        // page grid (PT tokens/page; PT > 0 whenever paged indexer
        // handles exist — ensure_indexer_pages refuses PT <= 0).  The
        // child keeps page groups [0, ceil(prefix_len / PT)) clamped to
        // what the parent holds; handles are (page, computing-layer,
        // rank)-ordered, so a group-boundary slice preserves the layout
        // ensure_indexer_pages resumes growth from (have = size / group).
        const int ik_pt =
            deps_.live_config->memory.kv_cache.indexer_k_page_size_tokens;
        const size_t ik_groups_avail = group > 0 ? src_ik.size() / group : 0;
        const size_t ik_want = (truncated && ik_pt > 0)
            ? (static_cast<size_t>(prefix_len)
               + static_cast<size_t>(ik_pt) - 1) / static_cast<size_t>(ik_pt)
            : ik_groups_avail;
        const size_t ik_take = std::min(ik_want, ik_groups_avail);
        const size_t ik_take_handles =
            (truncated && ik_pt > 0) ? ik_take * group : src_ik.size();
        dst_ik.reserve(ik_take_handles);
        for (size_t i = 0; i < ik_take_handles; ++i) {
            deps_.page_allocator->add_ref(src_ik[i]);
            dst_ik.push_back(src_ik[i]);
        }
        auto rollback_all = [&]() {
            for (auto& hh : dst_ik) deps_.page_allocator->free(hh);
            for (auto& hh : dst_handles)
                if (hh.page_idx >= 0) deps_.page_allocator->free(hh);
            if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
        };
        // R4a: same straddle rule as the KV split — a truncated child CoWs
        // its last kept group only when its first indexer append
        // (position prefix_len, page prefix_len / PT) lands INSIDE it.
        const bool ik_cow = !frozen && dst_ik.size() >= group
            && (!truncated || ik_pt <= 0
                || (prefix_len % static_cast<uint32_t>(ik_pt) != 0
                    && ik_take == ik_want));
        if (ik_cow) {
            const size_t first = dst_ik.size() - group;
            for (size_t i = first; i < dst_ik.size(); ++i) {
                try {
                    dst_ik[i] =
                        deps_.page_allocator->cow_copy(dst_ik[i], dst_id);
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

    // ── GF3.8: KDA state is COPY-ON-FORK, frozen holders INCLUDED ──
    // Refcount sharing is safe only between two parties that never step;
    // one side of every fork keeps stepping (a frozen holder's parent, a
    // live child), and a step mutates the whole slot in place — so every
    // fork takes a full ~slot_bytes D2D copy (~146 MiB fp32 at H=64/L=34,
    // ~0.1 ms at NVLink-class D2D; INV-PREFIX-CACHE-3 third cost class:
    // holder budget must include whole slots, which the boot sizing does).
    // The copy rides each GPU's kAttention stream, ordering it AFTER the
    // parent's last state write on that stream (the V4 side-tier copy
    // argument, verbatim — GF3.9 must keep KDA state kernels on the
    // kAttention stream).
    std::vector<memory::PageHandle> dst_kda;
    // GF3.12: a SPILLED holder's state lives in host RAM (kda_spill) —
    // the fork claims fresh child slots and H2D-restores the whole-slot
    // bytes (bit-exact fp32 round-trip, the checkpoint unit). Same
    // kAttention-stream ordering as the D2D arm: the child's first state
    // launch rides the same stream, so the copy is ordered before it.
    // The holder KEEPS its host copy — later hits fork from it again.
    //
    // P-29 step 24: a CHECKPOINT fork (kda_ckpt_fork above) restores from
    // the position-prefix_len host blob instead — same H2D arm, different
    // source bytes. It takes precedence over both the live slot and the
    // registered-length spill: the truncating child must stand at the
    // CHECKPOINT frontier, not the source's. The source KEEPS the blob
    // (later divergences fork from it again).
    const bool src_spilled = !src_kda_spill.empty();
    const std::vector<SequenceState::KdaSpillRank>* kda_host_src =
        kda_ckpt_fork ? &src_ckpt_it->second
                      : (src_spilled ? &src_kda_spill : nullptr);
    if (!src_kda_state.empty() || kda_host_src) {
        auto* pa = deps_.page_allocator;
        // TD-KDA-STATE-MAPPED-SLABS: handles are RANK-MAJOR [rank][unit]
        // (1 unit/rank carve, num_layers mapped). The copy is per UNIT —
        // same bytes, same order; a spilled holder's host blob is always
        // whole-slot layout, so unit u lives at offset u * unit_bytes.
        const int units = std::max(1, pa->kda_units_per_rank());
        const auto unit_bytes = static_cast<size_t>(pa->kda_unit_bytes());
        bool kda_ok = true;
        const size_t kda_ranks =
            kda_host_src ? kda_host_src->size()
                         : src_kda_state.size() / static_cast<size_t>(units);
        for (size_t r = 0; r < kda_ranks; ++r) {
            const int src_gpu = kda_host_src
                ? (*kda_host_src)[r].gpu_idx
                : src_kda_state[r * static_cast<size_t>(units)].gpu_idx;
            auto hs = pa->allocate_kda_state(src_gpu, dst_id);
            if (hs.empty()) { kda_ok = false; break; }
            const auto g = static_cast<size_t>(src_gpu);
            if (g < deps_.device_backends.size()
                && deps_.device_backends[g] && deps_.stream_manager) {
                auto* be = deps_.device_backends[g];
                be->set_device();
                void* stream = deps_.stream_manager->stream(
                    src_gpu, compute::StreamId::kAttention);
                for (size_t u = 0; u < hs.size(); ++u) {
                    if (kda_host_src)
                        be->memcpy_h2d_async(
                            hs[u].gpu_ptr,
                            static_cast<const char*>(
                                (*kda_host_src)[r].data()) + u * unit_bytes,
                            unit_bytes, stream);
                    else
                        be->memcpy_d2d_async(
                            hs[u].gpu_ptr,
                            src_kda_state[r * static_cast<size_t>(units)
                                          + u].gpu_ptr,
                            unit_bytes, stream);
                }
            }
            dst_kda.insert(dst_kda.end(), hs.begin(), hs.end());
        }
        if (!kda_ok) {
            for (auto& hh : dst_kda) deps_.page_allocator->free(hh);
            for (auto& hh : dst_ik) deps_.page_allocator->free(hh);
            for (auto& hh : dst_handles)
                if (hh.page_idx >= 0) deps_.page_allocator->free(hh);
            if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
            // 44z policy-failure poke: write_error's choke point.
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kKvPoolExhausted,
                        "seq_fork: exhausted KDA state pool cloning the "
                        "parent's slot — retryable, evict a prefix holder");
            return;
        }
    }

    // TD-PREFIX-FORK-COV (prefix-cache root cause, 2026-08-18): the DSA
    // indexer COVERAGE state machine (indexer_cov: next_pos high-water mark
    // + kUnset/kPaged/kDead mode) is per-seq and was not forked — the
    // child's first append at pos >= 1 read next_pos=0, was classified a GAP
    // and went PERMANENTLY kDead: every forked child silently decoded with
    // DENSE attention while a fresh sequence ran DSA sparse, flipping
    // near-tie tokens (different kernel accumulation over the same context)
    // and losing sparse perf for the child's whole life. Clone the parent's
    // coverage so the child's appends continue from the forked frontier.
    // (S4 deleted the legacy executor-arena mode; every live parent is
    // kPaged and its pages are refcount-shared with the child above.)
    IndexerCov dst_cov = src_cov;
    // R4a: a truncated child's coverage is exactly [0, min(parent
    // frontier, prefix_len)) — coverage is contiguous per position and
    // next_pos is the frontier, so the clamp is the whole truncation.  If
    // the parent covered LESS than prefix_len (mode died, or coverage
    // lagged), the child inherits the shorter frontier and its first
    // append at prefix_len is a GAP — the existing guard downgrades it to
    // dense (kDead) instead of scoring never-written rows: fail-closed,
    // no special case.  `mode` itself is never touched by truncation.
    if (truncated)
        dst_cov.next_pos = std::min(dst_cov.next_pos, prefix_len);
    // INV-DSA-EPOCH: the child draws its own rewind epoch — its seq_id
    // already keeps its keys distinct from the parent's, but a fresh epoch
    // also covers a recycled child id (and R4 truncating forks, whose child
    // content diverges from any earlier holder of the id at these positions).
    dst_cov.epoch = ++indexer_epoch_next_;

    const auto forked_count = static_cast<uint32_t>(dst_handles.size());
    // INV-SEQ-FORK-STATE enumeration point (fork-child construction): every
    // SequenceState member gets an explicit value or an explicit exclusion.
    {
        SequenceState st;
        auto& [kv_pages, forked, indexer_pages, indexer_cov, v4_tiers,
               indexer_reserved_tokens, kda_state, kda_next_pos,
               kda_spill, kda_ckpts, mtp_indexer_cov, rewind_run,
               kda_anchors] = st;
        (void)kda_spill;  // GF3.12: a fork child is live (or a fresh
                          // holder) — never born spilled; the state
                          // arrived in its own slot(s) above
        (void)kda_ckpts;  // P-29 step 24: a fork child is born without
                          // checkpoints — a FROZEN registration child
                          // INHERITS the parent's whole map after every
                          // rollback-prone step succeeds (the move at the
                          // completion seam below); a truncating hit
                          // child is live and captures its own during
                          // replay (the source keeps its blobs)
        // P-29 step 13: anchors are round-transient (never copied), but a fork
        // CHILD of a spec-armed boot needs its own anchor slots or its
        // rounds degrade to draftless. Best-effort claim (failure = empty
        // slots = graceful draftless; never fails the fork).
        // NOTE: test dst_kda (the claimed handles) — the kda_state binding
        // is only move-assigned further down this block.
        if (!dst_kda.empty() && deps_.live_config
            && deps_.live_config->speculation.enabled
            && deps_.live_config->speculation.method
                   == config::SpeculationMethodType::mtp
            && deps_.live_config->speculation.mtp.enabled) {
            std::string anchor_err;
            for (int k = 0; k < 2; ++k) {
                if (!claim_kda_state(dst_id, static_cast<int>(cmd.gpu_idx),
                                     kda_anchors.slots[k], anchor_err)) {
                    for (auto& av : kda_anchors.slots)
                        for (auto& hh : av) deps_.page_allocator->free(hh);
                    for (auto& av : kda_anchors.slots) av.clear();
                    spdlog::warn("seq_fork: child {} anchor claim failed "
                                 "({}) — speculative rounds degrade to "
                                 "draftless", dst_id, anchor_err);
                    break;
                }
            }
        }
        (void)src_rewind_run;   // P-29 step 13: step-transient — never forked
        (void)rewind_run;
        // P-29 step 13: MTP-layer coverage clones like the trunk coverage (the
        // MTP KV/indexer pages ride the same layer-major page copy), but
        // truncating forks reset it to the prefix like the trunk (the
        // MTP store lags the trunk, so min() is the honest frontier).
        mtp_indexer_cov = src_mtp_cov;
        // P-32 stage 1 BUG FIX (TD-MTP-COV-FROZEN-FORK): the clamp must
        // mirror the trunk clone's `if (truncated)` guard. A FROZEN
        // registration fork carries prefix_len == 0 (full refcount share —
        // `truncated` is false), and the unconditional step-13 clamp zeroed
        // every holder's MTP coverage: each prefix-hit child then dispatched
        // its first MTP row at pos == prefix_len against next_pos == 0, was
        // classified a GAP, and went PERMANENTLY dense on the MTP layer
        // (measured: indexer_dense_steps=481/leg on every prefix-hit spec
        // leg; dense MTP draft steps cost mean 5.2 ms vs 2.0 sparse at 8k).
        if (truncated && mtp_indexer_cov.next_pos > prefix_len)
            mtp_indexer_cov.next_pos = prefix_len;
        mtp_indexer_cov.epoch = dst_cov.epoch;
        kv_pages = std::move(dst_handles);
        forked = true;
        // GF3.8: the child's own state slot(s), D2D-copied above (frozen
        // and live forks alike — sharing is never safe for a mutate-in-
        // place slot).
        kda_state = std::move(dst_kda);
        // GF3.9: the child's KDA frontier == the parent's — the D2D above
        // copied the state exactly at these per-layer positions.
        // P-29 step 24: a CHECKPOINT fork restored the position-prefix_len
        // blob instead, so the child stands at the checkpoint frontier on
        // EVERY linear layer (uniform by the capture precondition) — the
        // INV-KDA-REWIND enforcement point must say so, or the replay's
        // first launch at prefix_len would be refused (or worse, a launch
        // at the parent's frontier would be silently accepted).
        if (kda_ckpt_fork)
            kda_next_pos.assign(
                static_cast<size_t>(
                    deps_.page_allocator->kda_layout().num_layers),
                prefix_len);
        else
            kda_next_pos = src_kda_next_pos;
        indexer_pages = std::move(dst_ik);
        indexer_cov = dst_cov;
        indexer_reserved_tokens = 0;  // granted below (post-repromote) on
                                      // a successful child reservation
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

    // R4a: a truncated child may have inherited a COLD straddling page
    // (hibernated-holder source with the boundary below its demoted
    // frontier).  The straddle is the truncated sequence's WRITE FRONTIER
    // — a LIVE child's first chunk write lands at prefix_len INSIDE it,
    // and a FROZEN truncated holder hands it to the NEXT fork's child
    // (the hit path full-forks the holder, whose frontier CoW skips
    // neutralized handles, so the child would trip the INV-KVT-2 cohort
    // check at its first chunk write — the R4a GPU smoke's leg-C
    // finding).  This is exactly INV-KVT-4's frontier rule ("the
    // write-frontier page stays hot so a hit-child's first chunk write
    // lands on the proven hot path") applied to the NEW mid-page
    // frontier, so lift it NOW for BOTH frozen and live truncated forks
    // through the proven cold-slot CoW-split machinery (INV-KVT-15):
    // repromote_for_rewind no-ops when the truncated frontier is already
    // <= prefix_len (page-aligned truncation included), and otherwise
    // re-promotes ONLY the straddling page into the new sequence's own
    // fresh VRAM copy (pages wholly inside the prefix stay COLD and are
    // read through the tiered materialize path, INV-KVT-13).  Failure is
    // VRAM capacity, not correctness: roll the whole fork back with a
    // RETRYABLE pool-exhausted error (the orchestrator's
    // fork-evict-retry seam keys on the category + "exhausted",
    // INV-IPC-ERRMSG-80).
    if (truncated && kv_tiering_
        && !kv_tiering_->repromote_for_rewind(dst_id, prefix_len)) {
        auto cit = sequences_.find(dst_id);
        if (cit != sequences_.end()) {
            if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
            for (auto& h : cit->second.kv_pages)
                if (h.page_idx >= 0) deps_.page_allocator->free(h);
            for (auto& h : cit->second.kda_state)  // GF3.8 (unreachable on
                deps_.page_allocator->free(h);     // KDA models: truncation
                                                   // is rejected above)
            if (!cit->second.indexer_pages.empty()) {
                for (auto& h : cit->second.indexer_pages)
                    deps_.page_allocator->free(h);
                for (auto& table : indexer_page_table_)
                    std::fill(table.begin(), table.end(), nullptr);
            }
            sequences_.erase(cit);
            invalidate_kv_meta();
            if (page_budget_.active_sequences > 0)
                --page_budget_.active_sequences;
            if (page_budget_.active_forks > 0) --page_budget_.active_forks;
        }
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kSeqFork,
                    "seq_fork: pool exhausted re-promoting truncated "
                    "child's straddling page");
        return;
    }

    // TD-INDEXER-NO-DENSE-FALLBACK (Route 1): the fork-time half of
    // reserve-at-admission — a HIT child (full or truncated fork from a
    // holder) is the other way a serving sequence is born, and its delta
    // prefill used to be exactly where lazy provisioning died mid-request
    // (the 2026-08-24 forked-25k-prefix incident). Grow the child's
    // indexer-K pages to the requested target NOW; failure rolls the whole
    // fork back with the RETRYABLE pool-exhaustion error the orchestrator's
    // fork-evict-retry seam already answers. Frozen holders pass 0 (they
    // never append; registration stays a pure refcount share). A child
    // whose inherited coverage is DEAD is REFUSED when a reservation was
    // requested — it could only ever serve dense, and the contract is
    // serve-sparse-or-refuse.
    uint32_t granted_reserve = 0;
    if (!frozen && cmd.seq_fork.reserve_tokens > 0
        && model_has_paged_indexer()) {
        auto fail_fork = [&](ipc::CmpErrorCategory cat, const char* msg) {
            auto cit = sequences_.find(dst_id);
            if (cit != sequences_.end()) {
                if (kv_tiering_) kv_tiering_->on_seq_free(dst_id);
                for (auto& h : cit->second.kv_pages)
                    if (h.page_idx >= 0) deps_.page_allocator->free(h);
                if (!cit->second.indexer_pages.empty()) {
                    for (auto& h : cit->second.indexer_pages)
                        deps_.page_allocator->free(h);
                    for (auto& table : indexer_page_table_)
                        std::fill(table.begin(), table.end(), nullptr);
                }
                // GF3.9 (leak found in the GF3.9 survey): the child's KDA
                // state slot(s) were D2D-claimed above — a failed child
                // reservation must release them like seq_free does, or a
                // glm5_next fork failure strands ~146 MiB per slot.
                for (auto& h : cit->second.kda_state)
                    deps_.page_allocator->free(h);
                sequences_.erase(cit);
                invalidate_kv_meta();
                if (page_budget_.active_sequences > 0)
                    --page_budget_.active_sequences;
                if (page_budget_.active_forks > 0)
                    --page_budget_.active_forks;
            }
            write_error(cmd.cmd_seq, cmd.gpu_idx, cat, msg);
        };
        const auto& child_cov = sequences_[dst_id].indexer_cov;
        if (child_cov.mode == IndexerSeqMode::kDead) {
            fail_fork(ipc::CmpErrorCategory::kSeqFork,
                      "seq_fork: child reservation refused — inherited "
                      "indexer coverage is DEAD (child would serve DENSE)");
            return;
        }
        const auto max_seq = static_cast<uint32_t>(
            deps_.live_config->serving.max_sequence_length);
        granted_reserve = std::min(cmd.seq_fork.reserve_tokens, max_seq);
        const int dcpr = deps_.dcp_executor ? deps_.dcp_executor->dcp_size()
                                            : 1;
        const auto r = grow_indexer_pages(dst_id, granted_reserve - 1, dcpr);
        if (r != IndexerPageResult::kOk) {
            spdlog::warn(
                "seq_fork: child {} REFUSED — indexer-K reservation for {} "
                "tokens {} (retryable={})", dst_id, granted_reserve,
                r == IndexerPageResult::kExhausted ? "exhausted the pool"
                                                   : "is unavailable",
                r == IndexerPageResult::kExhausted);
            if (r == IndexerPageResult::kExhausted) {
                // 44z policy-failure poke: fail_fork -> write_error's
                // kKvPoolExhausted choke point.
                fail_fork(ipc::CmpErrorCategory::kKvPoolExhausted,
                          "seq_fork: exhausted indexer-K pool at child "
                          "reservation — retryable, evict a prefix holder");
            }
            else
                fail_fork(ipc::CmpErrorCategory::kSeqFork,
                          "seq_fork: child indexer-K reservation "
                          "unavailable (beyond serving window)");
            return;
        }
        sequences_[dst_id].indexer_reserved_tokens = granted_reserve;
    }

    // P-29 step 24: at a FROZEN registration fork the parent's host KDA
    // checkpoints MOVE to the holder child — checkpoints belong to the
    // prefix entry, and the parent (the live request) is freed at end of
    // request while the holder answers future divergence hits. Done HERE,
    // after every rollback-prone step, so a failed fork leaves the parent's
    // blobs untouched (the rollbacks above never see them). All captured
    // positions are <= the parent frontier == the holder's registered
    // length (capture requires pos == the current uniform frontier), so no
    // clamp is needed. Global byte counter unchanged — ownership moved,
    // bytes did not. unordered_map emplace/erase invalidates no other
    // iterator, so src_it is still valid.
    if (frozen && !src_it->second.kda_ckpts.empty()) {
        auto dit = sequences_.find(dst_id);
        if (dit != sequences_.end()) {
            dit->second.kda_ckpts = std::move(src_it->second.kda_ckpts);
            src_it->second.kda_ckpts.clear();
            spdlog::info(
                "seq_fork: {} host KDA checkpoint(s) moved to frozen "
                "holder {} (P-29 step 24 prefix checkpoints)",
                dit->second.kda_ckpts.size(), dst_id);
        }
    }

    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, dst_id, forked_count, 0,
                         granted_reserve);
}

void CommandDispatcher::handle_seq_hibernate(const ipc::Command& cmd) {
    // R3 holder hibernation (TD-PREFIX-POOL-PRESSURE-EVICTS-THE-PRIZE): a
    // FROZEN prefix holder never steps, so window-driven demotion never
    // reaches it and whatever was hot at fork time stays VRAM-pinned by its
    // refs for the holder's whole life (~the retention window: measured
    // 5,100-5,200 pages/rank per deep holder on the GLM champion).  This
    // command demotes ALL of the sequence's demote-eligible kMain pages —
    // every hot page except the append-frontier logical group (kept hot so
    // a later fork's CoW split and the child's first k_append stay on the
    // proven hot path, INV-KVT-4) — through the exact after_attention
    // demotion machinery (stream-ordered D2H, refcount-aware free;
    // fair-share-cap-EXEMPT — the cap counts fork-family holders as
    // independent sequences — but pool-capacity fail-safe: full = pages
    // stay hot).  No-op success without a tiering manager
    // (V4 / untired arms) — the caller does not need to know the arch.
    const uint64_t seq_id = cmd.seq_hibernate.seq_id;
    const uint32_t kv_len = cmd.seq_hibernate.kv_len;
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFree,
                    "seq_hibernate: unknown seq_id");
        return;
    }
    if (kv_len == 0) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kSeqFree,
                    "seq_hibernate: kv_len required (holder KV coverage)");
        return;
    }
    uint32_t enqueued = 0;
    if (kv_tiering_) {
        const auto& pgs = it->second.kv_pages;
        const int L = kv_layers_ > 0 ? kv_layers_ : 1;
        const int num_logical = static_cast<int>(pgs.size()) / L;
        // Demote ONLY pages strictly below the holder's coverage-end page.
        // Pages at/after kv_len/page_size hold the write frontier and the
        // parent's over-allocation (windowed admission) — a hit-child's
        // first k_append lands there and needs them HOT; they also hold no
        // holder-covered row when kv_len is page-aligned (grid holders),
        // so demoting them would waste cold slots on garbage.
        const int PS = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
        const int frontier =
            std::min(static_cast<int>(kv_len) / PS,
                     num_logical > 0 ? num_logical - 1 : 0);
        if (num_logical > 0) {
            // S3 (tiering by slab): one whole-sequence sweep — all layers'
            // eligible pages collect into a single slab-grouped flush, so
            // the holder's cold token range leaves as contiguous slab runs
            // (same demote set + frontier rule as the per-layer loop this
            // replaces).
            enqueued = static_cast<uint32_t>(
                kv_tiering_->hibernate_seq(seq_id, pgs.data(), num_logical,
                                           frontier));
            // Reap already-complete copies now; the rest complete on the
            // next begin_layer/fork/free poll (steady serving traffic).
            kv_tiering_->poll_demotions();
        }
    }
    // GF3.12 (the GF3.8 deferred question, decided): a glm5_next holder's
    // dominant cost is its FULL KDA state slot (~146 MiB fp32/TP GPU —
    // INV-PREFIX-CACHE-3 third cost class), and a FROZEN holder's slot
    // content is FINAL (it never steps; arch_glm5_next refuses a step on a
    // slotless sequence loudly). So hibernation moves the whole-slot bytes
    // to HOST RAM — NUMA-local to each source GPU (NumaManager, the D2H
    // placement rule) with a heap fallback — and returns the VRAM slot(s)
    // to the pool: holders stop competing with live sequences for the pool
    // that clamps concurrency on this architecture. Bit-exact by the same
    // argument as the GF3.12 checkpoint (whole-slot fp32 byte round-trip;
    // replay from it is bit-identical, INV-KDA-CARRY); the D2H rides the
    // kAttention stream, ordered AFTER the registration fork's D2D that
    // produced this slot. A later hit's fork H2D-restores from the host
    // copy (the holder keeps it until seq_free). Fail-safe: any host
    // allocation or backend failure keeps the slot in VRAM (capacity,
    // never correctness). Ops kill switch: LS_KDA_HOLDER_SPILL=0.
    {
        const char* ksp = std::getenv("LS_KDA_HOLDER_SPILL");
        const bool kda_spill_on = !(ksp && ksp[0] == '0');
        auto& seq = it->second;
        if (kda_spill_on && !seq.kda_state.empty() && deps_.page_allocator
            && deps_.page_allocator->kda_state_slot_bytes() > 0) {
            const auto slot_bytes = static_cast<size_t>(
                deps_.page_allocator->kda_state_slot_bytes());
            // P-29 step 24: the gather loop moved to kda_gather_slot_to_host
            // (shared with D_CMD_KDA_CKPT) — byte-identical behavior.
            std::vector<SequenceState::KdaSpillRank> spill;
            const bool ok = kda_gather_slot_to_host(seq, spill);
            const size_t spill_ranks =
                seq.kda_state.size()
                / static_cast<size_t>(
                      std::max(1, deps_.page_allocator->kda_units_per_rank()));
            if (ok && spill.size() == spill_ranks) {
                for (auto& h : seq.kda_state)
                    deps_.page_allocator->free(h);
                seq.kda_state.clear();
                seq.kda_spill = std::move(spill);
                spdlog::info(
                    "seq_hibernate: seq {} KDA state spilled to host ({} "
                    "rank slot(s) x {:.1f} MiB; VRAM slot(s) returned to "
                    "the pool)", seq_id, seq.kda_spill.size(),
                    static_cast<double>(slot_bytes) / (1024.0 * 1024.0));
            } else {
                for (auto& r : spill)
                    if (r.buf.data && deps_.numa_manager)
                        deps_.numa_manager->free(r.buf);
                spdlog::warn(
                    "seq_hibernate: seq {} KDA state spill failed (host "
                    "alloc or backend unavailable) — holder keeps its "
                    "VRAM slot(s); capacity, not correctness", seq_id);
            }
        }
    }
    // TD-PREFIX-TIDY-COLD-SPILL: spill=1 takes the SECOND tiering hop —
    // the (already or just) hibernated holder's settled cold pages go to
    // one spill file and their pinned slots return to the pool.  status 2
    // = byte-cap refusal (the orchestrator evicts spilled holders —
    // deleting their files — and may retry); spilled page count rides the
    // reserved_tokens completion slot.
    uint32_t status = 0;
    uint32_t spilled = 0;
    if (cmd.seq_hibernate.spill && kv_tiering_) {
        kv_tiering_->drain_demotions();  // the cold set must be settled
        const int r = kv_tiering_->spill_seq(seq_id);
        if (r < 0) status = 2;
        else spilled = static_cast<uint32_t>(r);
    }
    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, enqueued, status,
                         spilled);
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

    // TD-KVT-COLD-FULL-HOT-WEDGE: one pressure sweep per ensure_pages call —
    // a second exhaustion after a sweep means the backlog is drained (or the
    // cold pool is the limit) and must surface as the retryable
    // kKvPoolExhausted class for the orchestrator's holder-eviction seam.
    bool pressure_swept = false;

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
            // headroom protects (INV-4.9f). S2: the single-GPU arm routes
            // through the sequence's position-major bump run.
            const auto try_alloc = [&]() {
                return replicated_kv
                    ? pa->allocate_replicated(seq_id, ts,
                                              static_cast<uint32_t>(l), pool,
                                              /*unreserved=*/true)
                    : pa->allocate_for_sequence(
                          gpu, pool, seq_id, static_cast<uint32_t>(l), ts,
                          ts + static_cast<uint32_t>(page_size),
                          /*unreserved=*/true);
            };
            auto h = try_alloc();
            if (!h && !pressure_swept && kv_tiering_
                && pool == memory::Pool::kMain) {
                // TD-KVT-COLD-FULL-HOT-WEDGE: a tiered sequence's hot
                // backlog (demotions skipped while the cold pool was full /
                // over budget) only drains at an after_attention boundary —
                // which needs the very step this exhaustion is blocking.
                // Demote it out-of-step, drain, and retry ONCE before
                // surfacing the retryable error.
                pressure_swept = true;
                if (tiering_pressure_reclaim(seq_id) > 0) h = try_alloc();
            }
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

int CommandDispatcher::tiering_pressure_reclaim(uint64_t seq_id) {
    if (!kv_tiering_) return 0;
    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    int enq = 0;
    const auto sweep = [&](uint64_t sid, const auto& st) {
        const auto& pgs = st.kv_pages;
        const int num_logical = static_cast<int>(pgs.size()) / L;
        if (num_logical > 0)
            enq += kv_tiering_->pressure_demote(sid, pgs.data(), num_logical);
    };
    // The requesting sequence carries the backlog in the measured wedge (one
    // live long prefill); other live tiered sequences are swept only when it
    // yields nothing (their own steps normally drain their own windows).
    if (auto it = sequences_.find(seq_id); it != sequences_.end())
        sweep(seq_id, it->second);
    if (enq == 0)
        for (const auto& [sid, st] : sequences_)
            if (sid != seq_id) sweep(sid, st);
    if (enq > 0) {
        // Block until the D2H copies land and free_page returned every
        // demoted VRAM page to the allocator (the caller retries the
        // allocation immediately after).
        kv_tiering_->drain_demotions();
        spdlog::info("KvTiering: kMain pressure sweep reclaimed {} "
                     "behind-window pages for seq {} "
                     "(TD-KVT-COLD-FULL-HOT-WEDGE)", enq, seq_id);
    }
    return enq;
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
// GF3.12: versions 3/4 = v1/v2 + a KDA recurrent-state section (glm5_next):
// SeqCkptKdaExt immediately after the header, then the unchanged KV +
// indexer bodies, then per-rank WHOLE-SLOT state blobs (rank order; slot =
// fp32 recurrent state + 3 conv rings per linear layer — the GF3.6/GF3.8
// checkpoint unit, self-contained and bit-exact on a byte round-trip).
// VERSIONING RULE (honest both ways): an old reader REJECTS v3/v4 at the
// version check — it can never silently drop the state; a new reader
// REFUSES to restore a state-carrying sequence from a v1/v2 file (no
// state to resume from — resuming on a zeroed state is silently wrong
// everywhere, INV-KDA-REWIND) and refuses v3/v4 onto a stateless
// sequence (config mismatch). Non-KDA models keep WRITING v1/v2
// byte-identically. V4 side-tier ring state (kSwa/kHca/LID + executor
// compressor rings) is in NO version of this format: snapshot/restore
// refuse on a sequence physically carrying V4 tiers rather than emit a
// checkpoint that resumes on garbage rings (TD-V4-CKPT-SIDE-TIERS).

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
/// GF3.12 (versions 3/4 only): KDA state section descriptor, written
/// immediately after SeqCkptHeader. All three fields are restore-time
/// exact-match requirements (a mismatch is a config mismatch: different
/// TP head shards or a different layer geometry cannot resume).
struct SeqCkptKdaExt {
    uint64_t slot_bytes = 0;  ///< PageAllocator::kda_state_slot_bytes()
    uint32_t ranks = 0;       ///< engaged TP GPUs (per-rank head shards)
    uint32_t frontier = 0;    ///< state frontier; must == token_count
};
}  // namespace

std::pair<int, int> CommandDispatcher::kda_seq_slots(uint64_t seq_id) const {
    const auto* st = find_seq(seq_id);
    if (!st) return {0, 0};
    return {static_cast<int>(st->kda_state.size()),
            static_cast<int>(st->kda_spill.size())};
}

std::vector<void*> CommandDispatcher::kda_seq_unit_ptrs(
        uint64_t seq_id) const {
    std::vector<void*> out;
    const auto* st = find_seq(seq_id);
    if (!st) return out;
    out.reserve(st->kda_state.size());
    for (const auto& h : st->kda_state) out.push_back(h.gpu_ptr);
    return out;
}

std::vector<int> CommandDispatcher::kda_seq_unit_gpus(
        uint64_t seq_id) const {
    std::vector<int> out;
    const auto* st = find_seq(seq_id);
    if (!st) return out;
    out.reserve(st->kda_state.size());
    for (const auto& h : st->kda_state) out.push_back(h.gpu_idx);
    return out;
}

uint32_t CommandDispatcher::kda_seq_frontier(uint64_t seq_id) const {
    const auto* st = find_seq(seq_id);
    if (!st || st->kda_next_pos.empty()) return UINT32_MAX;
    return st->kda_next_pos[0];
}

void CommandDispatcher::kda_test_set_frontier(uint64_t seq_id, uint32_t pos) {
    auto* st = find_seq(seq_id);
    if (!st || !deps_.page_allocator) return;
    st->kda_next_pos.assign(
        static_cast<size_t>(deps_.page_allocator->kda_layout().num_layers),
        pos);
}

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
    // GF3.12: V4 side-tier ring state is in NO version of this format —
    // a checkpoint of a tier-carrying sequence would restore onto zeroed
    // rings (silently wrong attention). Refuse loudly, physical check
    // (unit dispatchers flip configs mid-test) — TD-V4-CKPT-SIDE-TIERS.
    if (!it->second.v4_tiers.swa.empty())
        return fail("seq_snapshot: V4 side-tier ring state not in ckpt "
                    "format (TD-V4-CKPT-SIDE-TIERS)");
    // GF3.12 (the GF3.8 refusal, lifted): a state-carrying sequence
    // writes format v3/v4 — the whole-slot byte copy through
    // SequenceState::kda_state (or the holder's host spill) x
    // kda_state_slot_bytes(); self-contained, fp32 round-trip exact
    // (replay from it is bit-identical, INV-KDA-CARRY).
    const bool kda_seq = !it->second.kda_state.empty()
                      || !it->second.kda_spill.empty();
    if (kda_seq) {
        // The state must correspond EXACTLY to the claimed token_count:
        // every linear layer's frontier at count (a mid-superchunk or
        // stale-frontier snapshot would pair KV@count with state@other —
        // a checkpoint that lies). INV-KDA-REWIND anchor semantics.
        const auto& np = it->second.kda_next_pos;
        bool frontier_ok = !np.empty();
        for (uint32_t v : np) if (v != count) frontier_ok = false;
        if (!frontier_ok)
            return fail("seq_snapshot: KDA state frontier != token_count "
                        "(mid-step or never-stepped sequence)");
    }
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
    // TD-PREFIX-TIDY-COLD-SPILL: a spilled holder's cold bytes live on
    // disk — reload them so cold_page_host_ptr serves every demoted page.
    if (tiered_seq && kv_tiering_->seq_spilled(seq_id)
        && !kv_tiering_->unspill_seq(seq_id))
        return fail("seq_snapshot: spilled holder reload failed "
                    "(TD-PREFIX-TIDY-COLD-SPILL)");

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
    h.version = (sharded ? 2u : 1u) + (kda_seq ? 2u : 0u);
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
    int ihd = 0;
    int ik_n_computing = 1;
    int ik_entries = 0;     // stored rows per page (PT legacy, PT/kpool pooled)
    int ik_kpool = 1;       // tokens per stored entry (IndexPool, GF3.5)
    int64_t ik_bytes = 0;
    if (!seq_ik.empty() && deps_.live_config) {
        PT = deps_.live_config->memory.kv_cache.indexer_k_page_size_tokens;
        ihd = deps_.live_config->model.index_head_dim;
        // GF3.12 (found by the ckpt gate's byte diff): the page byte count
        // and layout are the ALLOCATOR's, not PT*(ihd+4) — on IndexPool
        // models (glm5_next) a page stores PT/kpool pooled entries plus the
        // in-progress pool's raw bf16 tail ([2][kpool][ihd]), NOT PT
        // per-token rows, and is ~4x SMALLER than the legacy formula: the
        // old arithmetic read (and restore WROTE) far past the page.
        // Legacy models: indexer_k_bytes_per_page == PT*(ihd+4) exactly,
        // so v1/v2 files stay byte-identical.
        model::ModelConfig mc(deps_.live_config->model);
        ik_bytes = memory::indexer_k_bytes_per_page(mc, *deps_.live_config);
        ik_entries = memory::indexer_k_entries_per_page(
            mc, *deps_.live_config);
        if (mc.has_index_pool())
            ik_kpool = deps_.live_config->model.index_kpool;
        for (size_t i = 0; i < seq_ik.size(); i += ik_group_stride)
            ik_reps.emplace_back(seq_ik[i].gpu_ptr, seq_ik[i].gpu_idx);
        // GF3.12 determinism + hygiene: pages past the covered range hold
        // whatever their previous physical tenant wrote — provisioned but
        // never appended (reserve-at-admission over-coverage). Writing
        // them would leak ANOTHER request's indexer bytes into the file
        // and make equal-state checkpoints byte-unequal. Keep only the
        // (page, computing-layer) groups covering [0, count); restore's
        // ensure_indexer_pages re-grows the reservation shape anyway.
        int n_computing = 0;
        for (uint8_t c : indexer_computes_) n_computing += c;
        ik_n_computing = std::max(1, n_computing);
        if (PT > 0) {
            const size_t cov_pages =
                (static_cast<size_t>(count) + PT - 1) / PT;
            const size_t cov_groups = cov_pages
                * static_cast<size_t>(ik_n_computing);
            if (ik_reps.size() > cov_groups) ik_reps.resize(cov_groups);
        }
    }
    h.ik_page_tokens = static_cast<uint32_t>(PT);
    h.ik_page_bytes = static_cast<uint64_t>(ik_bytes);
    h.ik_groups = static_cast<uint32_t>(ik_reps.size());

    SeqCkptKdaExt kext{};
    if (kda_seq) {
        kext.slot_bytes = static_cast<uint64_t>(
            deps_.page_allocator->kda_state_slot_bytes());
        // TD-KDA-STATE-MAPPED-SLABS: kda_state holds [rank][unit] handles
        // — the checkpoint counts RANKS (whole-slot blobs), identical
        // across modes so the FILE FORMAT never changes.
        kext.ranks = static_cast<uint32_t>(
            it->second.kda_state.empty()
                ? it->second.kda_spill.size()
                : it->second.kda_state.size()
                      / static_cast<size_t>(std::max(
                            1, deps_.page_allocator->kda_units_per_rank())));
        kext.frontier = count;
        if (kext.slot_bytes == 0 || kext.ranks == 0)
            return fail("seq_snapshot: KDA slot geometry unavailable");
    }

    std::FILE* f = std::fopen(path, "wb");
    if (!f) return fail("seq_snapshot: cannot open path");
    std::fwrite(&h, sizeof(h), 1, f);
    if (kda_seq) std::fwrite(&kext, sizeof(kext), 1, f);

    std::vector<std::byte> buf(static_cast<size_t>(stride));
    auto* base0 = static_cast<std::byte*>(kv_cache_base_ptrs_[0]);
    // GF3.12 determinism + hygiene: rows of the LAST logical page beyond
    // the covered token count were never appended by this sequence — they
    // hold the previous physical tenant's bytes (another request's KV).
    // Zero them in the file: no cross-request byte leak, and two
    // checkpoints of EQUAL sequence state are byte-EQUAL (the GPU ckpt
    // gate diffs whole files). Pages are row-major per token
    // (kv_bytes_per_page = bytes_per_token * page_size_tokens), so the
    // masked span is exactly [cov_rows * row_bytes, stride).
    const int64_t kv_row_bytes = stride / page_size;
    auto mask_kv_tail = [&](int j) {
        const int cov_rows = static_cast<int>(count)
                           - j * page_size;
        if (cov_rows < page_size)
            std::memset(buf.data()
                            + static_cast<size_t>(cov_rows) * kv_row_bytes,
                        0,
                        static_cast<size_t>(stride
                            - cov_rows * kv_row_bytes));
    };
    // GF3.12 (ckpt-gate byte diff, second find): nextn/MTP draft layers
    // (l >= num_hidden_layers) append KV/indexer rows ONLY during
    // speculation rounds — in any other run their pages hold recycled
    // physical-page history (another sequence's bytes). Draft state is
    // ADVISORY (the target verifies exactly; a cold draft only lowers
    // acceptance), so the checkpoint NEVER carries it: draft-layer blobs
    // are written as zeros, and a restored sequence re-warms its draft
    // KV through subsequent speculation. This also keeps equal-state
    // checkpoints byte-equal.
    const int hidden_layers = deps_.live_config
        ? deps_.live_config->model.num_hidden_layers : L;
    for (int j = 0; j < logical; ++j) {
        for (int l = 0; l < L; ++l) {
            if (l >= hidden_layers) {
                std::memset(buf.data(), 0, static_cast<size_t>(stride));
                std::fwrite(buf.data(), 1, static_cast<size_t>(stride), f);
                continue;
            }
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
                std::memcpy(buf.data(), cold, static_cast<size_t>(stride));
                mask_kv_tail(j);
                std::fwrite(buf.data(), 1, static_cast<size_t>(stride), f);
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
            mask_kv_tail(j);
            std::fwrite(buf.data(), 1, static_cast<size_t>(stride), f);
        }
    }
    if (ik_bytes > 0) {
        // Computing-ordinal -> layer map (groups are (page, computing-
        // layer)-ordered): needed to zero the draft (nextn) layers' groups
        // — same never-carried rule as the KV loop above.
        std::vector<int> ik_ordinal_layer;
        for (size_t l = 0; l < indexer_computes_.size(); ++l)
            if (indexer_computes_[l])
                ik_ordinal_layer.push_back(static_cast<int>(l));
        std::vector<std::byte> ikbuf(static_cast<size_t>(ik_bytes));
        for (size_t g = 0; g < ik_reps.size(); ++g) {
            const size_t ord = g % static_cast<size_t>(ik_n_computing);
            if (ord < ik_ordinal_layer.size()
                && ik_ordinal_layer[ord] >= hidden_layers) {
                std::memset(ikbuf.data(), 0, static_cast<size_t>(ik_bytes));
                std::fwrite(ikbuf.data(), 1,
                            static_cast<size_t>(ik_bytes), f);
                continue;
            }
            const auto& [p, gpu] = ik_reps[g];
            // Read from the GPU that HOLDS the page (owner under local mode;
            // rank 0's replica under replicated).
            auto* dev = deps_.device_backends[gpu];
            dev->set_device();
            dev->memcpy_d2h_async(ikbuf.data(), p,
                                  static_cast<size_t>(ik_bytes), nullptr);
            dev->synchronize_device();
            // GF3.12: mask the un-appended part of the page — same
            // determinism/hygiene rule as the KV tail above. Group order
            // is (page, computing-layer), so page = g / n_computing.
            // Layout [E x ihd FP8 | E x 4B scales | tail]: entry e is
            // WRITTEN only when its last token arrived, i.e. complete
            // entries in this page = cov_tokens / kpool (legacy kpool=1:
            // = covered rows). The pooled tail slot j (raw bf16 K + gate,
            // [2][kpool][ihd]) is written by the most recent token with
            // pos %% kpool == j — garbage only while cov_tokens < kpool.
            const int pg = static_cast<int>(
                g / static_cast<size_t>(ik_n_computing));
            const int cov_tok = std::min<int>(
                PT, static_cast<int>(count) - pg * PT);
            const int cov_e = cov_tok / ik_kpool;
            if (cov_e < ik_entries) {
                std::memset(ikbuf.data()
                                + static_cast<size_t>(cov_e) * ihd,
                            0,
                            static_cast<size_t>(ik_entries - cov_e) * ihd);
                std::memset(
                    ikbuf.data() + static_cast<size_t>(ik_entries) * ihd
                        + static_cast<size_t>(cov_e) * sizeof(float),
                    0,
                    static_cast<size_t>(ik_entries - cov_e)
                        * sizeof(float));
            }
            if (ik_kpool > 1 && cov_tok < ik_kpool) {
                const size_t tail0 = static_cast<size_t>(ik_entries) * ihd
                    + static_cast<size_t>(ik_entries) * sizeof(float);
                const size_t slot_b = static_cast<size_t>(ihd) * 2;  // bf16
                for (int plane = 0; plane < 2; ++plane)
                    std::memset(ikbuf.data() + tail0
                                    + (static_cast<size_t>(plane) * ik_kpool
                                       + cov_tok) * slot_b,
                                0,
                                static_cast<size_t>(ik_kpool - cov_tok)
                                    * slot_b);
            }
            std::fwrite(ikbuf.data(), 1, static_cast<size_t>(ik_bytes), f);
        }
    }
    if (kda_seq) {
        // Per-rank whole-slot state blobs, rank order. A spilled holder's
        // bytes are already host-resident; a live slot reads D2H on the
        // kAttention stream — ordered AFTER the sequence's last state
        // write (the GF3.8 stream contract), then synced.
        const auto slot_bytes = static_cast<size_t>(kext.slot_bytes);
        if (!it->second.kda_state.empty()) {
            // TD-KDA-STATE-MAPPED-SLABS: mapped handles GATHER into the
            // whole-slot blob at their per-layer offsets — the file bytes
            // are identical across modes (sbuf is value-initialized, so
            // an alignment tail is deterministic zero either way).
            const int units = std::max(
                1, deps_.page_allocator->kda_units_per_rank());
            const auto unit_bytes = static_cast<size_t>(
                deps_.page_allocator->kda_unit_bytes());
            const size_t ranks = it->second.kda_state.size()
                               / static_cast<size_t>(units);
            std::vector<std::byte> sbuf(slot_bytes);
            for (size_t r = 0; r < ranks; ++r) {
                const auto& sh0 = it->second.kda_state[
                    r * static_cast<size_t>(units)];
                const auto g = static_cast<size_t>(sh0.gpu_idx);
                if (g >= deps_.device_backends.size()
                    || !deps_.device_backends[g] || !sh0.gpu_ptr) {
                    std::fclose(f);
                    return fail("seq_snapshot: KDA slot backend "
                                "unavailable");
                }
                auto* be = deps_.device_backends[g];
                be->set_device();
                void* stream = deps_.stream_manager
                    ? deps_.stream_manager->stream(
                          sh0.gpu_idx, compute::StreamId::kAttention)
                    : nullptr;
                for (int u = 0; u < units; ++u) {
                    const auto& sh = it->second.kda_state[
                        r * static_cast<size_t>(units)
                        + static_cast<size_t>(u)];
                    be->memcpy_d2h_async(
                        sbuf.data() + static_cast<size_t>(u) * unit_bytes,
                        sh.gpu_ptr, unit_bytes, stream);
                }
                be->synchronize_device();
                std::fwrite(sbuf.data(), 1, slot_bytes, f);
            }
        } else {
            for (const auto& r : it->second.kda_spill)
                std::fwrite(r.data(), 1, slot_bytes, f);
        }
    }
    std::fclose(f);
    spdlog::info("seq_snapshot: seq {} @{} tokens -> {} ({} kv pages, {} ik "
                 "groups{})", seq_id, count, path, logical * L, h.ik_groups,
                 kda_seq ? ", + KDA state (v3/v4)" : "");
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
    // GF3.12: V4 side-tier ring state is in NO version of this format
    // (mirror of the seq_snapshot refusal; TD-V4-CKPT-SIDE-TIERS).
    if (!seq_it->second.v4_tiers.swa.empty())
        return fail("seq_restore: V4 side-tier ring state not in ckpt "
                    "format (TD-V4-CKPT-SIDE-TIERS)");
    // GF3.12: a hibernated (spilled) holder is frozen — restoring onto it
    // is not a meaningful operation (restore targets a freshly created
    // sequence). Refuse rather than guess where the state should land.
    if (!seq_it->second.kda_spill.empty())
        return fail("seq_restore: target is a spilled (hibernated) "
                    "holder — restore needs a fresh sequence");
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
        || h.version < 1 || h.version > 4) {
        std::fclose(f);
        return fail("seq_restore: bad header");
    }
    // GF3.12: versions 3/4 carry a KDA state section; 1/2 do not. Both
    // directions must refuse rather than resume wrong (the honest
    // versioning rule): a state-carrying sequence CANNOT resume from a
    // v1/v2 file (its recurrent state is the only memory of the linear
    // layers and is unrecoverable from KV, INV-KDA-REWIND — a zeroed
    // state is silently wrong everywhere), and a stateless sequence
    // cannot accept a v3/v4 file (nowhere for the state to land).
    const bool ckpt_kda = h.version >= 3;
    const bool seq_kda = !seq_it->second.kda_state.empty();
    if (seq_kda && !ckpt_kda) {
        std::fclose(f);
        return fail("seq_restore: ckpt carries no KDA state (v1/v2) — "
                    "a recurrent-state sequence cannot resume from it");
    }
    if (!seq_kda && ckpt_kda) {
        std::fclose(f);
        return fail("seq_restore: ckpt carries KDA state but the target "
                    "sequence has no state slot (config mismatch)");
    }
    SeqCkptKdaExt kext{};
    if (ckpt_kda) {
        if (std::fread(&kext, sizeof(kext), 1, f) != 1) {
            std::fclose(f);
            return fail("seq_restore: truncated KDA ext header");
        }
        const auto want_bytes = static_cast<uint64_t>(
            deps_.page_allocator->kda_state_slot_bytes());
        const size_t want_ranks = seq_it->second.kda_state.size()
            / static_cast<size_t>(std::max(
                  1, deps_.page_allocator->kda_units_per_rank()));
        if (kext.slot_bytes != want_bytes
            || kext.ranks != want_ranks
            || kext.frontier != h.token_count) {
            std::fclose(f);
            return fail("seq_restore: KDA geometry mismatch vs checkpoint "
                        "(slot bytes / TP ranks / frontier)");
        }
    }
    const int L = kv_layers_ > 0 ? kv_layers_ : 1;
    const int page_size = deps_.kv_page_size > 0 ? deps_.kv_page_size : 64;
    const int64_t stride = deps_.kv_cache_stride_block;
    const int dcp = deps_.dcp_executor ? deps_.dcp_executor->dcp_size() : 1;
    // KVS-2: the version's BASE mode is the KV placement the checkpoint
    // was taken under (1/3 = replicated, 2/4 = sequence-sharded).
    // Restoring across modes would place content on the wrong rank(s) —
    // reject.
    const bool sharded = kv_sharded_ && dcp >= 2;
    const bool ckpt_sharded = h.version == 2 || h.version == 4;
    if (h.kv_layers != static_cast<uint32_t>(L)
        || h.kv_page_size != static_cast<uint32_t>(page_size)
        || h.kv_stride_block != static_cast<uint64_t>(stride)
        || h.dcp_size != static_cast<uint32_t>(dcp)
        || ckpt_sharded != sharded) {
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
    int64_t ik_bytes_now = 0;
    if (deps_.live_config) {
        model::ModelConfig mc(deps_.live_config->model);
        ik_bytes_now =
            memory::indexer_k_bytes_per_page(mc, *deps_.live_config);
    }
    if (h.ik_groups > 0
        && (h.ik_page_tokens != static_cast<uint32_t>(PT_now)
            || h.ik_page_bytes != static_cast<uint64_t>(ik_bytes_now)
            || ensure_indexer_pages(seq_id, h.token_count - 1, 0, dcp)
                   != IndexerPageResult::kOk)) {
        seq_it->second.indexer_cov = {h.token_count, IndexerSeqMode::kDead,
                                      ++indexer_epoch_next_};
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
        // INV-DSA-EPOCH: a restore rewinds the sequence to checkpoint state
        // — the pre-restore life may have held different tokens at these
        // positions, so the restored coverage draws a fresh epoch.
        seq_it->second.indexer_cov = {h.token_count, IndexerSeqMode::kPaged,
                                      ++indexer_epoch_next_};
    }
    if (ckpt_kda) {
        // Per-rank whole-slot state upload, rank order, on the kAttention
        // stream: ordered AFTER the target's claim-time zero-memset and
        // BEFORE any later state kernel of this sequence (both ride the
        // same stream — the GF3.8 stream contract), then synced so the
        // file buffer can be released.
        const auto slot_bytes = static_cast<size_t>(kext.slot_bytes);
        // TD-KDA-STATE-MAPPED-SLABS: the file carries whole-slot blobs in
        // rank order regardless of mode; mapped handles SCATTER each
        // per-layer unit from its slot offset (the exact inverse of the
        // snapshot gather — fp32 round-trip stays bit-exact).
        const int units = std::max(
            1, deps_.page_allocator->kda_units_per_rank());
        const auto unit_bytes = static_cast<size_t>(
            deps_.page_allocator->kda_unit_bytes());
        const size_t want_ranks = seq_it->second.kda_state.size()
            / static_cast<size_t>(units);
        std::vector<std::byte> sbuf(slot_bytes);
        for (size_t r = 0; r < want_ranks; ++r) {
            if (std::fread(sbuf.data(), 1, slot_bytes, f) != slot_bytes) {
                std::fclose(f);
                return fail("seq_restore: truncated KDA state body");
            }
            const auto& sh0 = seq_it->second.kda_state[
                r * static_cast<size_t>(units)];
            const auto g = static_cast<size_t>(sh0.gpu_idx);
            if (g >= deps_.device_backends.size()
                || !deps_.device_backends[g] || !sh0.gpu_ptr) {
                std::fclose(f);
                return fail("seq_restore: KDA slot backend unavailable");
            }
            auto* be = deps_.device_backends[g];
            be->set_device();
            void* stream = deps_.stream_manager
                ? deps_.stream_manager->stream(
                      sh0.gpu_idx, compute::StreamId::kAttention)
                : nullptr;
            for (int u = 0; u < units; ++u) {
                const auto& sh = seq_it->second.kda_state[
                    r * static_cast<size_t>(units)
                    + static_cast<size_t>(u)];
                be->memcpy_h2d_async(
                    sh.gpu_ptr,
                    sbuf.data() + static_cast<size_t>(u) * unit_bytes,
                    unit_bytes, stream);
            }
            be->synchronize_device();
        }
        // The restored state stands exactly at the checkpoint frontier:
        // every linear layer absorbed [0, token_count) (snapshot enforced
        // it). Rebuild the per-layer INV-KDA-REWIND enforcement point so
        // the next launch must start there.
        seq_it->second.kda_next_pos.assign(
            static_cast<size_t>(
                deps_.page_allocator->kda_layout().num_layers),
            h.token_count);
    }
    std::fclose(f);

    invalidate_kv_meta();  // poison the dirty guard
    spdlog::info("seq_restore: seq {} <- {} @{} tokens ({} kv pages, {} ik "
                 "groups, dcp={}{})", seq_id, path, h.token_count,
                 h.logical_pages * L, h.ik_groups, dcp,
                 ckpt_kda ? ", + KDA state" : "");
    write_seq_completion(cmd.cmd_seq, cmd.gpu_idx, seq_id, h.token_count, 0);
}

// ── P-29 step 13 phase B: KDA anchor-and-replay (INV-KDA-REWIND) ──────────────────
// The ONLY legal KDA rewind is restore-from-anchor + forward replay. Two
// per-seq anchor slots (Pool::kKdaState claims — shared slab pool, no carve
// change) alternate by (pos / index_kpool) % 2, so pool-boundary anchors at
// consecutive boundaries never overwrite each other. Copies ride each rank's
// kAttention stream — ordered after the sequence's last state write and
// before its next state read, with no host sync.

bool CommandDispatcher::kda_anchor_copy_layer(SequenceState& st,
                                              int anchor_slot,
                                              int linear_ord, bool to_anchor) {
    auto* pa = deps_.page_allocator;
    const int units = std::max(1, pa->kda_units_per_rank());
    if (units <= 1) return false;  // carve mode: per-layer spans unsupported
    if (linear_ord < 0 || linear_ord >= units) return false;
    auto& anchor = st.kda_anchors.slots[anchor_slot];
    if (anchor.size() != st.kda_state.size()) return false;
    const auto unit_bytes = static_cast<size_t>(pa->kda_unit_bytes());
    const size_t ranks = st.kda_state.size() / static_cast<size_t>(units);
    for (size_t r = 0; r < ranks; ++r) {
        const size_t u = r * static_cast<size_t>(units)
                       + static_cast<size_t>(linear_ord);
        const int g = st.kda_state[u].gpu_idx;
        if (g < 0 || static_cast<size_t>(g) >= deps_.device_backends.size()
            || !deps_.device_backends[g] || !deps_.stream_manager)
            return false;
        auto* be = deps_.device_backends[g];
        be->set_device();
        void* stream = deps_.stream_manager->stream(
            g, compute::StreamId::kAttention);
        void* dst = to_anchor ? anchor[u].gpu_ptr : st.kda_state[u].gpu_ptr;
        void* src = to_anchor ? st.kda_state[u].gpu_ptr : anchor[u].gpu_ptr;
        if (!dst || !src) return false;
        be->memcpy_d2d_async(dst, src, unit_bytes, stream);
    }
    return true;
}

bool CommandDispatcher::kda_anchor_copy_all(SequenceState& st,
                                            uint64_t seq_id, int anchor_slot,
                                            bool to_anchor, const char** why) {
    auto* pa = deps_.page_allocator;
    const int units = std::max(1, pa->kda_units_per_rank());
    if (units <= 1) {
        *why = "KDA anchors require mapped state (carve mode unsupported)";
        return false;
    }
    auto& anchor = st.kda_anchors.slots[anchor_slot];
    if (to_anchor && anchor.empty()) {
        // Lazy claim: mirror the live slot's rank layout.
        const size_t ranks =
            st.kda_state.size() / static_cast<size_t>(units);
        for (size_t r = 0; r < ranks; ++r) {
            const int g =
                st.kda_state[r * static_cast<size_t>(units)].gpu_idx;
            auto hs = pa->allocate_kda_state(g, seq_id);
            if (hs.empty()) {
                for (auto& h : anchor) pa->free(h);
                anchor.clear();
                // 44z poke rides the caller's write_error (the "exhausted"
                // message maps to kKvPoolExhausted — the choke point).
                *why = "exhausted KDA state pool claiming an anchor slot "
                       "— retryable, evict a prefix holder";
                return false;
            }
            anchor.insert(anchor.end(), hs.begin(), hs.end());
        }
    }
    if (anchor.size() != st.kda_state.size()) {
        *why = "KDA anchor slot layout mismatch";
        return false;
    }
    const int num_lin = units;
    for (int ord = 0; ord < num_lin; ++ord)
        if (!kda_anchor_copy_layer(st, anchor_slot, ord, to_anchor)) {
            *why = "KDA anchor copy failed (device/stream missing)";
            return false;
        }
    return true;
}

void CommandDispatcher::handle_kda_snapshot(const ipc::Command& cmd) {
    const auto& p = cmd.kda_anchor;
    auto* st = find_seq(p.seq_id);
    if (!st || st->kda_state.empty()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_snapshot: unknown sequence or no KDA state");
        return;
    }
    // The anchor captures ONE uniform frontier: every linear layer must
    // have absorbed exactly [0, pos) (never snapshot mid-sweep).
    for (uint32_t f : st->kda_next_pos)
        if (f != p.pos) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kComputeValidation,
                        "kda_snapshot: per-layer frontier not uniform at "
                        "pos (mid-sweep snapshot refused)");
            return;
        }
    const int kp = std::max(
        1, deps_.live_config ? deps_.live_config->model.index_kpool : 1);
    // Slot 0 = pool-boundary anchors (always-legal restore targets);
    // slot 1 = per-round anchors (legal only while their pool is the
    // frontier pool). Keeps the two roles from evicting each other.
    const int slot = (p.pos % kp == 0) ? 0 : 1;
    const char* why = nullptr;
    if (!kda_anchor_copy_all(*st, p.seq_id, slot, /*to_anchor=*/true, &why)) {
        const bool pool = why && std::strstr(why, "exhausted");
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    pool ? ipc::CmpErrorCategory::kKvPoolExhausted
                         : ipc::CmpErrorCategory::kComputeValidation, why);
        return;
    }
    st->kda_anchors.pos[slot] = p.pos;
    write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                             /*layer=*/0, /*status=*/0,
                             /*host_buf_offset=*/0,
                             /*data_bytes=*/static_cast<uint32_t>(slot));
}

void CommandDispatcher::handle_kda_restore(const ipc::Command& cmd) {
    const auto& p = cmd.kda_anchor;
    auto* st = find_seq(p.seq_id);
    if (!st || st->kda_state.empty()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_restore: unknown sequence or no KDA state");
        return;
    }
    int slot = -1;
    for (int k = 0; k < 2; ++k)
        if (st->kda_anchors.pos[k] == p.pos
            && !st->kda_anchors.slots[k].empty())
            slot = k;
    if (slot < 0) {
        // INV-KDA-REWIND: a rewind without an anchor at the target throws,
        // never approximates.
        spdlog::error("kda_restore: seq {} pos {} — anchors [{} @{}, {} "
                      "@{}]", p.seq_id, p.pos,
                      st->kda_anchors.slots[0].size(),
                      st->kda_anchors.pos[0],
                      st->kda_anchors.slots[1].size(),
                      st->kda_anchors.pos[1]);
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_restore: no anchor recorded at pos "
                    "(INV-KDA-REWIND: un-anchored rewind refused)");
        return;
    }
    const char* why = nullptr;
    if (!kda_anchor_copy_all(*st, p.seq_id, slot, /*to_anchor=*/false,
                             &why)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation, why);
        return;
    }
    // Roll every linear layer's INV-KDA-REWIND enforcement point back to
    // the anchor: the next launch must start exactly there.
    if (st->kda_next_pos.empty())
        st->kda_next_pos.assign(
            static_cast<size_t>(
                deps_.page_allocator->kda_layout().num_layers), 0);
    for (auto& f : st->kda_next_pos) f = p.pos;
    // Any LATER anchor captured state that absorbed now-rejected tokens —
    // its byte content is stale for the new token history. Invalidate.
    for (int k = 0; k < 2; ++k)
        if (k != slot && st->kda_anchors.pos[k] != SequenceState::KdaAnchors::kNone
            && st->kda_anchors.pos[k] > p.pos)
            st->kda_anchors.pos[k] = SequenceState::KdaAnchors::kNone;
    write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                             /*layer=*/0, /*status=*/0,
                             /*host_buf_offset=*/0,
                             /*data_bytes=*/static_cast<uint32_t>(slot));
}

// ── P-29 step 24: host-RAM KDA prefix checkpoints (LS_KDA_PREFIX_CKPT) ──

bool CommandDispatcher::kda_gather_slot_to_host(
        const SequenceState& seq,
        std::vector<SequenceState::KdaSpillRank>& out) {
    // Extracted verbatim from the GF3.12 hibernate spill (the proven
    // bit-exact path): whole-slot layout ALWAYS — carve mode copies the
    // slot in one D2H per unit, mapped mode GATHERS the per-layer units
    // at their slot offsets (u * unit_bytes); blob bytes are identical
    // across modes (TD-KDA-STATE-MAPPED-SLABS). NUMA-local to each source
    // GPU with heap fallback; the slot-alignment tail is ZEROED so blobs
    // are bit-comparable; the D2H rides kAttention, ordered after the
    // producing writes on that stream, then a device sync fences the host
    // bytes. Failure releases every claimed NumaBuffer and returns false
    // (capacity, never correctness).
    out.clear();
    if (seq.kda_state.empty() || !deps_.page_allocator
        || deps_.page_allocator->kda_state_slot_bytes() <= 0)
        return false;
    const auto slot_bytes = static_cast<size_t>(
        deps_.page_allocator->kda_state_slot_bytes());
    const int units =
        std::max(1, deps_.page_allocator->kda_units_per_rank());
    const auto unit_bytes = static_cast<size_t>(
        deps_.page_allocator->kda_unit_bytes());
    const size_t ranks = seq.kda_state.size() / static_cast<size_t>(units);
    bool ok = true;
    for (size_t sr = 0; sr < ranks; ++sr) {
        const auto& h0 = seq.kda_state[sr * static_cast<size_t>(units)];
        SequenceState::KdaSpillRank r;
        r.gpu_idx = h0.gpu_idx;
        if (deps_.numa_manager) {
            try {
                r.buf = deps_.numa_manager->allocate_for_gpu(
                    slot_bytes, h0.gpu_idx);
            } catch (const std::bad_alloc&) { /* heap fallback */ }
        }
        if (!r.buf.data) {
            try { r.heap.resize(slot_bytes); }
            catch (const std::bad_alloc&) { ok = false; break; }
        }
        // Determinism: any slot-alignment tail past the logical units
        // must be ZERO bytes, not uninitialized host memory (the blob
        // feeds bit-compared checkpoints).
        {
            const size_t logical =
                static_cast<size_t>(units) * unit_bytes;
            if (logical < slot_bytes && r.data())
                std::memset(static_cast<char*>(r.data()) + logical, 0,
                            slot_bytes - logical);
        }
        const auto g = static_cast<size_t>(h0.gpu_idx);
        if (g < deps_.device_backends.size()
            && deps_.device_backends[g] && h0.gpu_ptr) {
            auto* be = deps_.device_backends[g];
            be->set_device();
            void* stream = deps_.stream_manager
                ? deps_.stream_manager->stream(
                      h0.gpu_idx, compute::StreamId::kAttention)
                : nullptr;
            for (int u = 0; u < units; ++u) {
                const auto& h = seq.kda_state[
                    sr * static_cast<size_t>(units)
                    + static_cast<size_t>(u)];
                be->memcpy_d2h_async(
                    static_cast<char*>(r.data())
                        + static_cast<size_t>(u) * unit_bytes,
                    h.gpu_ptr, unit_bytes, stream);
            }
            be->synchronize_device();
        } else { ok = false; break; }
        out.push_back(std::move(r));
    }
    if (!ok || out.size() != ranks) {
        for (auto& r : out)
            if (r.buf.data && deps_.numa_manager)
                deps_.numa_manager->free(r.buf);
        out.clear();
        return false;
    }
    return true;
}

void CommandDispatcher::handle_kda_ckpt(const ipc::Command& cmd) {
    // D_CMD_KDA_CKPT (P-29 step 24, GF3.12 realized): capture ONE
    // position-keyed, uncompressed, host-RAM KDA checkpoint of a LIVE
    // sequence's whole state slot. Preconditions are refused LOUDLY
    // (CMP_ERROR, the orchestrator's capture-tripwire counter must read
    // 0): the capture point must be a positive multiple of 64
    // (INV-KDA-CARRY bitwise grid — also KV-page- and indexer-pool-
    // aligned, so a later truncating fork at it never straddles anything)
    // AND the current UNIFORM per-layer frontier (never mid-sweep — the
    // handle_kda_snapshot discipline verbatim). Host-allocation failure
    // is CAPACITY: completes with status 1 and no checkpoint (serving
    // continues; the orchestrator simply has no reuse point here).
    // A duplicate position completes idempotently with status 0 and
    // data_bytes 0 (already held; nothing captured).
    const auto& p = cmd.kda_anchor;
    auto* st = find_seq(p.seq_id);
    if (!st || st->kda_state.empty()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_ckpt: unknown sequence or no live KDA state");
        return;
    }
    if (p.pos == 0 || p.pos % 64 != 0) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_ckpt: pos must be a positive multiple of 64 "
                    "(INV-KDA-CARRY grid)");
        return;
    }
    if (st->kda_next_pos.empty()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kComputeValidation,
                    "kda_ckpt: sequence has never stepped (no frontier)");
        return;
    }
    for (uint32_t f : st->kda_next_pos)
        if (f != p.pos) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kComputeValidation,
                        "kda_ckpt: per-layer frontier not uniform at pos "
                        "(mid-sweep capture refused)");
            return;
        }
    if (st->kda_ckpts.count(p.pos)) {
        write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                                 /*layer=*/0, /*status=*/0,
                                 /*host_buf_offset=*/0, /*data_bytes=*/0);
        return;
    }
    std::vector<SequenceState::KdaSpillRank> blob;
    if (!kda_gather_slot_to_host(*st, blob)) {
        spdlog::warn("kda_ckpt: seq {} pos {} capture failed (host alloc "
                     "or backend unavailable) — no checkpoint; capacity, "
                     "not correctness", p.seq_id, p.pos);
        write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                                 /*layer=*/0, /*status=*/1,
                                 /*host_buf_offset=*/0, /*data_bytes=*/0);
        return;
    }
    const auto slot_bytes = static_cast<size_t>(
        deps_.page_allocator->kda_state_slot_bytes());
    const size_t bytes = slot_bytes * blob.size();
    kda_ckpt_host_bytes_ += bytes;
    st->kda_ckpts.emplace(p.pos, std::move(blob));
    spdlog::info("kda_ckpt: seq {} checkpoint @{} captured ({} rank "
                 "slot(s), {:.1f} MiB host; seq total {} ckpt(s))",
                 p.seq_id, p.pos, st->kda_ckpts.at(p.pos).size(),
                 static_cast<double>(bytes) / (1024.0 * 1024.0),
                 st->kda_ckpts.size());
    write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                             /*layer=*/0, /*status=*/0,
                             /*host_buf_offset=*/0,
                             /*data_bytes=*/static_cast<uint32_t>(bytes));
}

}  // namespace layerstorm::daemon
