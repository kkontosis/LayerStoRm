// PinnedExpertArena — per-NUMA-node pinned, slab-managed expert arena (P-24).

#include "core/memory/pinned_expert_arena.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/cuda_hardware_query.h"
#include "core/memory/arena_cache.h"
#include "core/memory/arena_loader.h"
#include "core/memory/host_pool_sizing.h"
#include "core/memory/numa_manager.h"
#include "model/weight_pipeline/prepacked_source.h"

namespace layerstorm::memory {

// ── PinnedNodeArena ──────────────────────────────────────────────────────────

PinnedNodeArena::PinnedNodeArena(NumaManager& numa, int numa_node,
                                 size_t slot_size_bytes, size_t num_slots,
                                 int share_degree, size_t extra_scratch_bytes,
                                 bool defer_register, ArenaBackingMode backing,
                                 NumaBuffer adopted)
    : numa_(numa),
      numa_node_(numa_node),
      share_degree_(share_degree),
      slot_size_bytes_(slot_size_bytes) {
    if (slot_size_bytes_ == 0)
        throw std::runtime_error("PinnedNodeArena: slot_size_bytes must be > 0");
    if (num_slots == 0)
        throw std::runtime_error(
            "PinnedNodeArena: zero slots for node " + std::to_string(numa_node) +
            " (insufficient per-node RAM budget? INV-4.12f)");

    // General shared scratch reserved at the FRONT of the slab (before slot 0).
    // Page-round so slot 0 stays page-aligned (the arena base is page-aligned by
    // NumaManager; a page-rounded scratch keeps every slot page-aligned too —
    // matters for O_DIRECT slot reads). 0 => byte-for-byte the prior layout.
    if (extra_scratch_bytes > 0) {
        const size_t page = 4096;
        scratch_bytes_ = (extra_scratch_bytes + page - 1) & ~(page - 1);
    }

    total_bytes_ = scratch_bytes_ + slot_size_bytes_ * num_slots;

    // 1. One big arena slab, mbind'd to this node. Backing (P-24b):
    //    kPrivate       — MAP_ANONYMOUS, pages fault lazily (classic P-24).
    //    kSharedCreate  — fresh MAP_SHARED memfd (fd → arena holder later).
    //    kAdopt         — a holder-received segment; content preserved.
    switch (backing) {
        case ArenaBackingMode::kPrivate:
            buf_ = numa_.allocate_on_node(total_bytes_, numa_node_);
            break;
        case ArenaBackingMode::kSharedCreate:
            buf_ = numa_.allocate_on_node_shared(
                total_bytes_, numa_node_,
                ("ls-arena-n" + std::to_string(numa_node_)).c_str());
            break;
        case ArenaBackingMode::kAdopt:
            if (!adopted.data || adopted.size < total_bytes_) {
                const size_t got = adopted.size;
                numa_.free(adopted);  // don't leak the mapping/counters
                throw std::runtime_error(
                    "PinnedNodeArena: adopted segment for node " +
                    std::to_string(numa_node_) + " too small (" +
                    std::to_string(got) + " < " +
                    std::to_string(total_bytes_) + ")");
            }
            buf_ = adopted;   // ownership moves in; freed via numa_.free in dtor
            break;
    }
    base_ = buf_.data;
    if (!base_) {
        throw std::runtime_error(
            "PinnedNodeArena: allocation failed for node " +
            std::to_string(numa_node_));
    }

    slots_.resize(num_slots);

    // 2. Page-lock the arena. When `defer_register` is set the caller will
    //    prefault the pages NUMA-local (in parallel across nodes) and then call
    //    finalize_registration() — registration on resident pages is ~6× cheaper
    //    than on fresh pages (cudaHostRegister faults serially under a per-mm
    //    lock). See finalize_registration() and PinnedExpertArena's prefault.
    if (!defer_register)
        finalize_registration();
}

void PinnedNodeArena::finalize_registration() {
    if (registered_ || !base_) return;
    // ONE cudaHostRegister(Portable) over the whole arena — Portable so every
    // GPU context can DMA it (shared node serves multiple GPUs, P-24).
    auto t0 = std::chrono::steady_clock::now();
    int err = core::host_register_pinned_portable(base_, buf_.size);
    auto t1 = std::chrono::steady_clock::now();
    if (err != 0) {
        // Do NOT free here: registration can run concurrently with a preload that
        // holds slot pointers into base_ (TD-INIT-OVERLAP). Leave the arena
        // intact and let the dtor reclaim it once everyone has finished.
        throw std::runtime_error(
            "PinnedNodeArena: cudaHostRegister(Portable) failed for node " +
            std::to_string(numa_node_) + " (" +
            std::to_string(buf_.size / 1073741824.0) +
            " GB), cudaError " + std::to_string(err));
    }
    registered_ = true;

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double gb = buf_.size / 1073741824.0;
    spdlog::info("PinnedNodeArena: node {} — {:.1f} GB, {} slots, share_degree={}, "
                 "scratch={} MB, registered in {:.0f} ms ({:.0f} ms/GB)",
                 numa_node_, gb, slots_.size(), share_degree_,
                 scratch_bytes_ / 1048576, secs * 1e3,
                 gb > 0 ? secs * 1e3 / gb : 0.0);
}

PinnedNodeArena::~PinnedNodeArena() {
    if (registered_ && base_) {
        core::host_unregister_pinned(base_);
        registered_ = false;
    }
    if (buf_.data)
        numa_.free(buf_);
    base_ = nullptr;
}

void* PinnedNodeArena::slot_ptr(size_t i) const {
    // Slot region starts AFTER the front-of-slab scratch reservation.
    return static_cast<char*>(base_) + scratch_bytes_ + i * slot_size_bytes_;
}

void* PinnedNodeArena::resolve(ExpertKey key) const {
    auto it = index_.find(key);
    if (it == index_.end()) return nullptr;
    const Slot& s = slots_[it->second];
    if (!s.ready) return nullptr;
    return slot_ptr(it->second);
}

void* PinnedNodeArena::reserve(ExpertKey key, ExpertKey* evicted_out) {
    if (evicted_out) *evicted_out = kNoEvictedKey;
    auto it = index_.find(key);
    if (it != index_.end()) {
        // Already resident — LRU-touch and return existing slot.
        slots_[it->second].lru_tick = ++lru_clock_;
        return slot_ptr(it->second);
    }

    size_t idx = pick_slot();
    if (idx >= slots_.size())
        return nullptr;  // every slot pinned in-flight (momentarily full)

    Slot& s = slots_[idx];
    if (s.occupied) {
        // Evicting a resident slot — drop its index entry, report the victim.
        if (evicted_out) *evicted_out = s.key;
        index_.erase(s.key);
    }
    // INV-ARENA-CACHE-ORDER: the slot is about to hold NEW content — kill its
    // persistence record BEFORE the caller writes the first byte.
    if (cache_) cache_->on_reuse(numa_node_, idx);
    s.key = key;
    s.occupied = true;
    s.ready = false;
    s.loading = false;   // J-1: caller flips via mark_loading() if it submits async.
    s.lru_tick = ++lru_clock_;
    // inflight is already 0 (pick_slot only returns inflight==0 slots).
    index_[key] = idx;
    return slot_ptr(idx);
}

size_t PinnedNodeArena::pick_slot() {
    // Prefer an unoccupied slot.
    for (size_t i = 0; i < slots_.size(); ++i)
        if (!slots_[i].occupied) return i;

    // Otherwise evict the LRU slot with no in-flight DMA AND no async load in
    // flight (multi-GPU safe: inflight counts all GPUs' outstanding H2Ds; J-1:
    // a slot whose file→slot read is still running must not be reused, or its
    // worker would write into a slot now owned by another expert).
    size_t best = slots_.size();
    uint64_t best_tick = UINT64_MAX;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].inflight.load(std::memory_order_acquire) != 0) continue;
        if (slots_[i].loading) continue;   // J-1: pinned against eviction while loading
        if (slots_[i].lru_tick < best_tick) {
            best_tick = slots_[i].lru_tick;
            best = i;
        }
    }
    return best;  // slots_.size() if all in-flight or loading
}

void PinnedNodeArena::mark_loading(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    slots_[it->second].loading = true;
    // P-24b: stamp LOADING (identity recorded, not adoptable until ACTIVE).
    if (cache_) cache_->on_load_start(numa_node_, it->second, key);
}

void PinnedNodeArena::clear_loading(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    slots_[it->second].loading = false;
    // P-24b: the load failed — the slot's bytes are undefined; kill the record.
    if (cache_) cache_->on_reuse(numa_node_, it->second);
}

bool PinnedNodeArena::is_loading(ExpertKey key) const {
    auto it = index_.find(key);
    return it != index_.end() && slots_[it->second].loading;
}

void PinnedNodeArena::mark_ready(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    auto& s = slots_[it->second];
    // TD-GOLDEN bug #6: run the post-fill hook exactly once per fill, on the
    // not-ready -> ready transition (reformats NVFP4 scales in the slot).
    const bool transition = !s.ready;
    if (transition && post_fill_)
        post_fill_(key, slot_ptr(it->second));
    s.ready = true;
    s.loading = false;   // J-1: load complete.
    // P-24b: the slot's bytes are final (incl. any post_fill transform) —
    // publish the full record as ACTIVE (INV-ARENA-CACHE-ORDER).
    if (transition && cache_) cache_->on_load_ready(numa_node_, it->second, key);
}

bool PinnedNodeArena::adopt_slot(ExpertKey key, size_t slot_idx) {
    if (slot_idx >= slots_.size()) return false;
    if (slots_[slot_idx].occupied) return false;
    if (index_.count(key)) return false;   // key already resident elsewhere
    Slot& s = slots_[slot_idx];
    s.key = key;
    s.occupied = true;
    s.ready = true;      // content preserved across runs — no load, no post_fill
    s.loading = false;
    s.lru_tick = ++lru_clock_;
    index_[key] = slot_idx;
    return true;
}

bool PinnedNodeArena::is_ready(ExpertKey key) const {
    auto it = index_.find(key);
    return it != index_.end() && slots_[it->second].ready;
}

bool PinnedNodeArena::evict(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    Slot& s = slots_[it->second];
    // Never drop a slot an H2D is reading (inflight) or an async fill is
    // writing (loading). Single-threaded wrt acquire/reserve (daemon
    // thread), so this check cannot race a NEW pin; a concurrent release
    // from a completion callback only lowers the count (harmless).
    if (s.loading) return false;
    if (s.inflight.load(std::memory_order_acquire) != 0) return false;
    s.occupied = false;
    s.ready = false;
    index_.erase(it);
    // Intentionally NO cache stamp: the slot's bytes are intact, so its
    // ACTIVE record remains truthful (try_adopt's "free cell still holds
    // it" contract). reserve() stamps EMPTY before any overwrite.
    return true;
}

void PinnedNodeArena::touch(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    slots_[it->second].lru_tick = ++lru_clock_;
}

void PinnedNodeArena::acquire_inflight(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    slots_[it->second].inflight.fetch_add(1, std::memory_order_acq_rel);
}

void PinnedNodeArena::release_inflight(ExpertKey key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    int prev = slots_[it->second].inflight.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        // Underflow guard — restore and warn (a release without a matching
        // acquire is a caller bug).
        slots_[it->second].inflight.fetch_add(1, std::memory_order_acq_rel);
        spdlog::warn("PinnedNodeArena: release_inflight underflow for expert "
                     "({},{}) on node {}", key.layer_idx, key.expert_idx,
                     numa_node_);
    }
}

int PinnedNodeArena::inflight(ExpertKey key) const {
    auto it = index_.find(key);
    if (it == index_.end()) return 0;
    return slots_[it->second].inflight.load(std::memory_order_acquire);
}

// ── PinnedExpertArena (set of per-node arenas) ───────────────────────────────

std::vector<ArenaNodePlan> PinnedExpertArena::plan_geometry(
        NumaManager& numa, size_t slot_size_bytes, size_t total_arena_bytes,
        const config::PinHostExpertPoolSizingConfig& sizing,
        const config::CrossNodeSpillConfig& spill) {
    const auto& nodes = numa.gpu_attached_nodes();
    if (nodes.empty())
        throw std::runtime_error(
            "PinnedExpertArena: no GPU-attached NUMA nodes (cannot place arenas)");
    if (slot_size_bytes == 0)
        throw std::runtime_error("PinnedExpertArena: slot_size_bytes must be > 0");

    // Cross-node spill weights (node → weight>0). Empty/disabled → no-op.
    std::unordered_map<int, int> spill_weight;
    if (spill.enabled)
        for (const auto& sn : spill.nodes)
            if (sn.weight > 0) spill_weight[sn.node] = sn.weight;

    // share_degree[node] = number of GPUs whose numa_node maps to it. The shared
    // node (P-22 share_degree=2) gets one arena serving both GPUs (no split).
    std::unordered_map<int, int> share_degree;
    for (int n : nodes) share_degree[n] = 0;
    for (int g = 0; g < numa.num_gpus(); ++g) {
        int node = numa.gpu_numa_node(g);
        auto sit = share_degree.find(node);
        if (sit != share_degree.end()) ++sit->second;
    }
    for (int n : nodes)
        if (share_degree[n] < 1) share_degree[n] = 1;

    // Plan arenas on the UNION of GPU-attached nodes and spill nodes (deterministic
    // order). GPU-node budget is clamped to total/num_gpu_nodes (never exceed the
    // routed set); spill nodes are EXTRA warm capacity, sized by spill.sizing_*
    // with no /total clamp (a GPU-less spill node can be sized generously).
    std::set<int> union_nodes(nodes.begin(), nodes.end());
    for (const auto& [n, w] : spill_weight) union_nodes.insert(n);

    const size_t per_node_target = total_arena_bytes / nodes.size();
    config::PinHostExpertPoolSizingConfig spill_sizing;
    spill_sizing.mode = spill.sizing_mode;
    spill_sizing.value = spill.sizing_value;
    // TD-NUMA-HBM-BANKS: per-node overrides let the small CPU-less HBM spill nodes
    // use fraction_free (safe against their tiny free capacity) while big RAM spill
    // nodes keep the global fraction_total. resolve_node_budget_bytes checks per_node
    // first, so an override wins over sizing_mode/value for that node.
    spill_sizing.per_node = spill.per_node;

    std::vector<ArenaNodePlan> plans;
    plans.reserve(union_nodes.size());
    for (int n : union_nodes) {
        const bool is_spill = spill_weight.count(n) != 0;
        size_t node_budget;
        if (is_spill) {
            size_t cap = resolve_node_budget_bytes(spill_sizing, numa, n);
            node_budget = (cap == 0) ? per_node_target : cap;  // no /total clamp
        } else {
            size_t cap = resolve_node_budget_bytes(sizing, numa, n);
            if (cap == 0) cap = per_node_target;
            node_budget = std::min(per_node_target, cap);
        }
        size_t num_slots = node_budget / slot_size_bytes;
        if (num_slots == 0) {
            throw std::runtime_error(
                "PinnedExpertArena: node " + std::to_string(n) +
                (is_spill ? " spill" : "") + " budget (" +
                std::to_string(node_budget / 1048576) + " MB) < one slot (" +
                std::to_string(slot_size_bytes / 1048576) +
                " MB) — raise " + (is_spill ? "cross_node_spill.sizing_value"
                                            : "pin_host_expert_pool_sizing.value") +
                " or free RAM (INV-4.12f)");
        }
        int sd = share_degree.count(n) ? share_degree[n] : 1;  // spill-only → 1
        plans.push_back({n, is_spill, num_slots, sd});
    }
    return plans;
}

PinnedExpertArena::PinnedExpertArena(
        NumaManager& numa, size_t slot_size_bytes, size_t total_arena_bytes,
        const config::PinHostExpertPoolSizingConfig& sizing,
        const config::CrossNodeSpillConfig& spill,
        size_t extra_scratch_bytes, bool defer_registration,
        ArenaBacking* backing)
    : numa_(numa) {
    // Per-node parameters: compute + validate SERIALLY first (cheap arithmetic),
    // so the zero-slot error keeps its deterministic message/order before any
    // expensive pinning begins. kAdopt (P-24b): the STORED geometry
    // (backing->adopt_plans) is authoritative — plan_geometry resolves budgets
    // from CURRENT free RAM, which drifts run-to-run (and could even throw on
    // a momentarily tight node), while the adopted segments already exist.
    const bool adopting = backing &&
                          backing->mode == ArenaBackingMode::kAdopt;
    const std::vector<ArenaNodePlan> plans =
        adopting ? backing->adopt_plans
                 : plan_geometry(numa, slot_size_bytes, total_arena_bytes,
                                 sizing, spill);
    if (adopting && plans.empty())
        throw std::runtime_error(
            "PinnedExpertArena: kAdopt backing requires adopt_plans");

    // Cross-node spill weights, recomputed for spill_order_ (selection order).
    std::unordered_map<int, int> spill_weight;
    if (spill.enabled)
        for (const auto& sn : spill.nodes)
            if (sn.weight > 0) spill_weight[sn.node] = sn.weight;

    const ArenaBackingMode mode =
        backing ? backing->mode : ArenaBackingMode::kPrivate;

    // Build arenas in THREE phases to make page-locking ~3.7× faster.
    // cudaHostRegister faults every page resident as it pins, and that faulting
    // serializes on a per-mm lock — so registering fresh anonymous arenas costs
    // ~240 ms/GB and does NOT parallelize across nodes (measured). Instead:
    //   (1) allocate every arena (cheap lazy mmap+mbind, defer_register),
    //   (2) PREFAULT all arenas' pages concurrently via plain NUMA-local writes
    //       (read-mode mmap_lock → scales across nodes AND threads-per-node, hits
    //       memory bandwidth), then
    //   (3) finalize_registration() on now-resident pages (~40 ms/GB, cheap).
    // Measured on this box (382 GB, 4 nodes): ~240 → ~65 ms/GB end-to-end. This
    // is pure host-memory pinning — no disk involved, so no storage-tier gating.

    // (1) Allocate (no register). mmap+mbind is cheap; keep it serial+ordered so
    // an allocation failure throws deterministically before any pinning.
    // kAdopt (P-24b): consume the holder-received segment for each node — the
    // engine validated the plan against the stored geometry BEFORE calling.
    for (const auto& p : plans) {
        NumaBuffer adopted{};
        if (mode == ArenaBackingMode::kAdopt) {
            auto it = backing->adopted.find(p.node);
            if (it == backing->adopted.end())
                throw std::runtime_error(
                    "PinnedExpertArena: no adopted segment for planned node " +
                    std::to_string(p.node));
            adopted = it->second;
            it->second = NumaBuffer{};  // ownership moves to the node arena
        }
        arenas_[p.node] = std::make_unique<PinnedNodeArena>(
            numa_, p.node, slot_size_bytes, p.num_slots, p.share_degree,
            extra_scratch_bytes, /*defer_register=*/true, mode,
            std::move(adopted));
    }

    // (2) Parallel NUMA-local prefault. All nodes at once (cross-node bandwidth
    // adds up), several threads per node (one memory controller saturates around
    // ~8 threads). memset(0) preserves the zeroed anon-page semantics; the preload
    // overwrites real slot bytes afterward.
    // kAdopt: SKIPPED — the pages are already resident with the preserved slot
    // bytes; a memset would destroy exactly the content we attached to keep.
    if (mode != ArenaBackingMode::kAdopt) {
        unsigned hw = std::thread::hardware_concurrency();
        const size_t kThreadsPerNode =
            std::max<size_t>(1, std::min<size_t>(8, hw ? hw / arenas_.size() : 1));
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ft;
        ft.reserve(arenas_.size() * kThreadsPerNode);
        size_t total_gb_x = 0;
        for (const auto& [node, arena] : arenas_) {
            char* b = static_cast<char*>(arena->base());
            const size_t total = arena->total_bytes();
            total_gb_x += total;
            const size_t chunk = total / kThreadsPerNode;
            for (size_t t = 0; t < kThreadsPerNode; ++t) {
                char* base = b + t * chunk;
                const size_t len = (t == kThreadsPerNode - 1) ? total - t * chunk
                                                              : chunk;
                ft.emplace_back([this, node, base, len] {
                    numa_.pin_current_thread_to_node(node);
                    std::memset(base, 0, len);
                });
            }
        }
        for (auto& t : ft) t.join();
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double gb = total_gb_x / 1073741824.0;
        spdlog::info("PinnedExpertArena: prefaulted {:.1f} GB across {} node(s) "
                     "in {:.1f} s ({:.1f} GB/s, {} thread(s)/node)",
                     gb, arenas_.size(), secs, secs > 0 ? gb / secs : 0.0,
                     kThreadsPerNode);
    }

    // Spill candidates sorted DESC by weight (equal weights broken at selection).
    for (const auto& [n, w] : spill_weight) spill_order_.push_back({n, w});
    std::sort(spill_order_.begin(), spill_order_.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (!spill_order_.empty())
        spdlog::info("PinnedExpertArena: cross-node spill enabled — {} candidate "
                     "node(s)", spill_order_.size());

    // (3) Register each arena (cheap now — pages resident). When `defer_registration`
    // the caller runs register_all() later, overlapped with the io_uring preload
    // (disjoint hardware). Otherwise register here (prior single-shot behaviour).
    if (!defer_registration) register_all();

    spdlog::info("PinnedExpertArena: {} node arena(s), {} total slots, "
                 "{:.1f} GB pinned{}", arenas_.size(), total_slots(),
                 total_pinned_bytes() / 1073741824.0,
                 defer_registration ? " (registration deferred)" : "");
}

void PinnedExpertArena::register_all() {
    // One thread PER NODE arena: registration of RESIDENT pages (the ctor
    // prefaults, or an adopted store is resident by construction) pins under
    // the mm read lock and parallelizes across nodes — serial was the warm-
    // attach floor (~82 s for 504 GB; the P-24 "does not parallelize" finding
    // applied to FRESH pages, whose faults serialize under the write lock,
    // and those are prefaulted first now anyway). Safe concurrently with
    // preload(): finalize_registration touches only each arena's {base, buf,
    // registered} — never the slot index/LRU that preload mutates — and on
    // failure it throws WITHOUT freeing (a concurrent preload's slot pointers
    // stay valid; the dtor reclaims the arena after both finish). Errors are
    // collected and the first (by ascending node id) is rethrown for a
    // deterministic message.
    std::vector<std::pair<int, PinnedNodeArena*>> order;
    order.reserve(arenas_.size());
    for (auto& [node, arena] : arenas_) order.push_back({node, arena.get()});
    std::sort(order.begin(), order.end());

    std::vector<std::exception_ptr> errs(order.size());
    std::vector<std::thread> threads;
    threads.reserve(order.size());
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < order.size(); ++i) {
        threads.emplace_back([&, i] {
            try {
                order[i].second->finalize_registration();
            } catch (...) {
                errs[i] = std::current_exception();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (auto& e : errs)
        if (e) std::rethrow_exception(e);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double gb = total_pinned_bytes() / 1073741824.0;
    spdlog::info("PinnedExpertArena: registered {} node arena(s) in parallel — "
                 "{:.1f} GB in {:.1f} s ({:.0f} ms/GB effective)",
                 order.size(), gb, secs, gb > 0 ? secs * 1e3 / gb : 0.0);
}

PinnedNodeArena* PinnedExpertArena::arena_for(ExpertKey key) {
    int node = numa_.expert_home_node(key.expert_idx);
    if (node < 0) return nullptr;
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

const PinnedNodeArena* PinnedExpertArena::arena_for(ExpertKey key) const {
    int node = numa_.expert_home_node(key.expert_idx);
    if (node < 0) return nullptr;
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

PinnedNodeArena* PinnedExpertArena::arena_for_gpu(int gpu) {
    int node = numa_.gpu_numa_node(gpu);   // INV-MoE-NUMA: consuming GPU's node
    if (node < 0) return nullptr;
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

const PinnedNodeArena* PinnedExpertArena::arena_for_gpu(int gpu) const {
    int node = numa_.gpu_numa_node(gpu);
    if (node < 0) return nullptr;
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

// ── Location index (cross-node spill, Stage 1) ───────────────────────────────
// A spilled expert lives on neither its home node nor the computing GPU's node,
// so all facade state-mutators consult `location_` FIRST and route to the node
// that actually holds the slot; `gpu`/home routing is only a fallback for a key
// not yet placed.

PinnedNodeArena* PinnedExpertArena::located_arena(ExpertKey key) {
    auto it = location_.find(key);
    if (it == location_.end()) return nullptr;
    auto a = arenas_.find(it->second);
    return a == arenas_.end() ? nullptr : a->second.get();
}

const PinnedNodeArena* PinnedExpertArena::located_arena(ExpertKey key) const {
    auto it = location_.find(key);
    if (it == location_.end()) return nullptr;
    auto a = arenas_.find(it->second);
    return a == arenas_.end() ? nullptr : a->second.get();
}

int PinnedExpertArena::location_node(ExpertKey key) const {
    auto it = location_.find(key);
    return it == location_.end() ? -1 : it->second;
}

// ── Epoch-latched bank snapshot (INV-REEF-BANK) ──────────────────────────────

void PinnedExpertArena::publish_bank_snapshot() {
    auto snap =
        std::make_shared<const std::unordered_map<ExpertKey, int>>(location_);
    {
        std::lock_guard<std::mutex> lk(bank_snap_mu_);
        bank_snap_ = std::move(snap);
    }
    bank_epoch_.fetch_add(1, std::memory_order_release);
}

std::shared_ptr<const std::unordered_map<ExpertKey, int>>
PinnedExpertArena::bank_snapshot() const {
    std::lock_guard<std::mutex> lk(bank_snap_mu_);
    return bank_snap_;
}

// ── M3b online placement migration (INV-ARENA-MIGRATE) ───────────────────────

bool PinnedExpertArena::evict_key(ExpertKey key) {
    auto lit = location_.find(key);
    if (lit == location_.end()) return false;
    auto ait = arenas_.find(lit->second);
    if (ait == arenas_.end() || !ait->second->evict(key)) return false;
    const int old_node = lit->second;
    location_.erase(lit);
    note_loc(key, old_node, -1, 'E');
    return true;
}

void* PinnedExpertArena::migrate_begin(ExpertKey key, int target_node) {
    PinnedNodeArena* src = located_arena(key);
    if (!src || src->node() == target_node) return nullptr;
    if (!src->is_ready(key)) return nullptr;  // mid-fill or absent: not movable
    auto tit = arenas_.find(target_node);
    if (tit == arenas_.end()) return nullptr;
    PinnedNodeArena* tgt = tit->second.get();
    // Extend-only (like reserve_on_node): the caller frees a victim first.
    if (!tgt->has_free_slot()) return nullptr;
    if (tgt->resident(key)) return nullptr;  // stale half-migration guard
    // Pin the source slot for the whole migration window: pick_slot skips
    // inflight>0 slots, so `key` stays located+ready at the source and every
    // facade path (resolve / reserve_for_fill fast path / host_source)
    // keeps routing to the intact copy.
    src->acquire_inflight(key);
    ExpertKey evicted = kNoEvictedKey;
    void* dst = tgt->reserve(key, &evicted);
    if (!dst) {  // unreachable given has_free_slot() + single-threaded daemon
        src->release_inflight(key);
        return nullptr;
    }
    if (evicted != kNoEvictedKey) {  // paranoia
        const int en = location_node(evicted);
        location_.erase(evicted);
        note_loc(evicted, en, -1, 'E');
    }
    tgt->mark_loading(key);  // LOADING record; non-evictable target slot
    return dst;
}

PinnedExpertArena::MigrateCommit PinnedExpertArena::migrate_commit(
        ExpertKey key, int target_node) {
    PinnedNodeArena* src = located_arena(key);
    auto tit = arenas_.find(target_node);
    if (!src || tit == arenas_.end() || !tit->second->resident(key)) {
        // Contract violation (commit without begin) — nothing safe to do.
        spdlog::warn("PinnedExpertArena::migrate_commit: no begun migration "
                     "for expert ({},{}) → node {}", key.layer_idx,
                     key.expert_idx, target_node);
        return MigrateCommit::kRetry;
    }
    // inflight == 1 ⇔ only OUR migration pin remains: no demand H2D is
    // reading the source slot, and none can start inside this daemon tick.
    if (src->inflight(key) > 1) return MigrateCommit::kRetry;
    PinnedNodeArena* tgt = tit->second.get();
    tgt->mark_ready(key);        // ACTIVE record at the target (post_fill runs)
    src->release_inflight(key);  // migration pin off
    src->evict(key);             // source slot free; record intact (see evict)
    const int old_node = location_node(key);
    location_[key] = target_node;  // the flip: all routing now → target
    note_loc(key, old_node, target_node, 'C');
    // INV-REEF-BANK: a migrator commit is a placement-epoch boundary —
    // republish the bank snapshot (daemon thread; only once consumers have
    // armed the first epoch, so non-consuming boots pay nothing).
    if (bank_epoch_.load(std::memory_order_relaxed) > 0)
        publish_bank_snapshot();
    return MigrateCommit::kCommitted;
}

void PinnedExpertArena::migrate_abort(ExpertKey key, int target_node) {
    auto tit = arenas_.find(target_node);
    if (tit != arenas_.end()) {
        PinnedNodeArena* tgt = tit->second.get();
        if (tgt->resident(key) && !tgt->is_ready(key)) {
            tgt->clear_loading(key);  // record EMPTY — bytes are undefined
            tgt->evict(key);          // free the target slot
        }
    }
    if (PinnedNodeArena* src = located_arena(key)) src->release_inflight(key);
}

PinnedNodeArena* PinnedExpertArena::locate(ExpertKey key) {
    return located_arena(key);
}
const PinnedNodeArena* PinnedExpertArena::locate(ExpertKey key) const {
    return located_arena(key);
}

void* PinnedExpertArena::resolve(ExpertKey key) const {
    if (const PinnedNodeArena* a = located_arena(key)) return a->resolve(key);
    const PinnedNodeArena* a = arena_for(key);
    return a ? a->resolve(key) : nullptr;
}

bool PinnedExpertArena::is_ready(ExpertKey key) const {
    if (const PinnedNodeArena* a = located_arena(key)) return a->is_ready(key);
    const PinnedNodeArena* a = arena_for(key);
    return a && a->is_ready(key);
}

void* PinnedExpertArena::reserve(ExpertKey key) {
    // Home placement (used by preload): route to the holding node if located,
    // else the home arena; maintain the location index (drop any evicted key).
    PinnedNodeArena* a = located_arena(key);
    int node = location_node(key);
    if (!a) { a = arena_for(key); node = numa_.expert_home_node(key.expert_idx); }
    if (!a) return nullptr;
    ExpertKey evicted = kNoEvictedKey;
    void* slot = a->reserve(key, &evicted);
    if (slot) {
        if (evicted != kNoEvictedKey) {
            const int en = location_node(evicted);
            location_.erase(evicted);
            note_loc(evicted, en, -1, 'E');
        }
        const int prev = location_node(key);
        location_[key] = node;
        if (prev != node) note_loc(key, prev, node, 'R');
    }
    return slot;
}

void PinnedExpertArena::set_post_fill(
        std::function<void(ExpertKey, void*)> hook) {
    for (auto& [node, a] : arenas_)
        if (a) a->set_post_fill(hook);
}

// ── P-24b persistence (arena holder + ArenaCache) ────────────────────────────

void PinnedExpertArena::set_cache(ArenaCache* cache) {
    cache_ = cache;
    for (auto& [node, a] : arenas_)
        if (a) a->set_cache(cache);
}

bool PinnedExpertArena::adopt_ready(ExpertKey key, int node, size_t slot_idx) {
    auto it = arenas_.find(node);
    if (it == arenas_.end()) return false;
    if (location_.count(key)) return false;  // resident elsewhere
    if (!it->second->adopt_slot(key, slot_idx)) return false;
    location_[key] = node;
    note_loc(key, -1, node, 'R');
    return true;
}

void* PinnedExpertArena::try_adopt(ExpertKey key) {
    if (!cache_) return nullptr;
    const auto hit = cache_->lookup(key);
    if (!hit) return nullptr;
    auto it = arenas_.find(hit->node);
    if (it == arenas_.end()) return nullptr;
    if (it->second->slot_occupied(hit->slot)) return nullptr;  // bytes gone
    if (!adopt_ready(key, hit->node, hit->slot)) return nullptr;
    return it->second->resolve(key);
}

int PinnedExpertArena::node_backing_fd(int node) const {
    const PinnedNodeArena* a = node_arena(node);
    return a ? a->backing_fd() : -1;
}

size_t PinnedExpertArena::node_backing_bytes(int node) const {
    const PinnedNodeArena* a = node_arena(node);
    return a ? a->backing_bytes() : 0;
}

size_t PinnedExpertArena::slab_bytes(size_t slot_size_bytes, size_t num_slots,
                                     size_t extra_scratch_bytes) {
    // Mirror PinnedNodeArena's layout math (page-rounded front scratch) and
    // NumaManager::align_to_page (system page size).
    const size_t kScratchPage = 4096;
    const size_t scratch = extra_scratch_bytes
        ? ((extra_scratch_bytes + kScratchPage - 1) & ~(kScratchPage - 1))
        : 0;
    size_t total = scratch + slot_size_bytes * num_slots;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (total == 0) total = 1;
    return (total + page - 1) & ~(page - 1);
}

void PinnedExpertArena::mark_ready(ExpertKey key) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = arena_for(key);
    if (a) a->mark_ready(key);
}

void PinnedExpertArena::mark_loading(ExpertKey key, int gpu) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = (gpu >= 0) ? arena_for_gpu(gpu) : arena_for(key);
    if (a) a->mark_loading(key);
}

void PinnedExpertArena::clear_loading(ExpertKey key, int gpu) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = (gpu >= 0) ? arena_for_gpu(gpu) : arena_for(key);
    if (a) a->clear_loading(key);
}

bool PinnedExpertArena::is_loading(ExpertKey key, int gpu) const {
    const PinnedNodeArena* a = located_arena(key);
    if (!a) a = (gpu >= 0) ? arena_for_gpu(gpu) : arena_for(key);
    return a && a->is_loading(key);
}

void PinnedExpertArena::mark_ready(ExpertKey key, int gpu) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = (gpu >= 0) ? arena_for_gpu(gpu) : arena_for(key);
    if (a) a->mark_ready(key);
}

void PinnedExpertArena::acquire_inflight(ExpertKey key) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = arena_for(key);
    if (a) a->acquire_inflight(key);
}

void PinnedExpertArena::release_inflight(ExpertKey key) {
    PinnedNodeArena* a = located_arena(key);
    if (!a) a = arena_for(key);
    if (a) a->release_inflight(key);
}

std::vector<int> PinnedExpertArena::spill_candidates_in_order(int exclude) {
    // spill_order_ is sorted DESC by weight. Within an equal-weight group, order
    // by (fewest-occupied, round-robin tick, node) — deterministic, no rand/time.
    // Higher weight is always preferred until full; equal weights interleave.
    std::vector<int> out;
    size_t i = 0;
    while (i < spill_order_.size()) {
        int w = spill_order_[i].second;
        std::vector<int> group;
        size_t j = i;
        while (j < spill_order_.size() && spill_order_[j].second == w) {
            int n = spill_order_[j].first;
            if (n != exclude && arenas_.count(n)) group.push_back(n);
            ++j;
        }
        std::sort(group.begin(), group.end(), [&](int a, int b) {
            size_t oa = arenas_.at(a)->occupied(), ob = arenas_.at(b)->occupied();
            if (oa != ob) return oa < ob;
            uint64_t ra = rr_counter_[a], rb = rr_counter_[b];
            if (ra != rb) return ra < rb;
            return a < b;
        });
        for (int n : group) { out.push_back(n); ++rr_counter_[n]; }
        i = j;
    }
    return out;
}

void* PinnedExpertArena::reserve_for_fill(ExpertKey key, int gpu, int* out_node,
                                          bool extend_only) {
    if (out_node) *out_node = -1;
    // Already placed → reserve on that node (LRU-touch, no move).
    if (PinnedNodeArena* a = located_arena(key)) {
        ExpertKey ev = kNoEvictedKey;
        void* s = a->reserve(key, &ev);
        if (s && out_node) *out_node = location_node(key);
        return s;
    }
    // Candidate order: computing GPU's local node (tier-1), then weighted spill.
    int local = numa_.gpu_numa_node(gpu);
    std::vector<int> cands;
    if (local >= 0 && arenas_.count(local)) cands.push_back(local);
    for (int n : spill_candidates_in_order(local)) cands.push_back(n);
    if (cands.empty()) {  // no GPU-local arena (gpu<0 / unmapped) → home arena
        int hn = numa_.expert_home_node(key.expert_idx);
        if (hn >= 0 && arenas_.count(hn)) cands.push_back(hn);
    }
    // Phase A — EXTEND: first candidate with a FREE slot (don't evict yet).
    for (int n : cands) {
        PinnedNodeArena* a = arenas_.at(n).get();
        if (!a->has_free_slot()) continue;
        ExpertKey ev = kNoEvictedKey;
        if (void* s = a->reserve(key, &ev)) {
            if (ev != kNoEvictedKey) {
                const int en = location_node(ev);
                location_.erase(ev);
                note_loc(ev, en, -1, 'E');
            }
            location_[key] = n;
            note_loc(key, -1, n, 'R');
            if (out_node) *out_node = n;
            return s;
        }
    }
    // extend_only (preload): never evict — a slot just preloaded must not be
    // dropped to make room for another. Return null so the caller stops/skips.
    if (extend_only) return nullptr;
    // Phase B — EVICT: no free slot anywhere; evict the weighted-best node's LRU.
    for (int n : cands) {
        PinnedNodeArena* a = arenas_.at(n).get();
        ExpertKey ev = kNoEvictedKey;
        if (void* s = a->reserve(key, &ev)) {
            if (ev != kNoEvictedKey) {
                const int en = location_node(ev);
                location_.erase(ev);
                note_loc(ev, en, -1, 'E');
            }
            location_[key] = n;
            note_loc(key, -1, n, 'R');
            if (out_node) *out_node = n;
            return s;
        }
    }
    return nullptr;  // every candidate slot pinned in-flight
}

void* PinnedExpertArena::reserve_on_node(ExpertKey key, int node) {
    // Already placed → reserve on the holding node (LRU-touch, no move) —
    // mirrors reserve_for_fill's located fast path.
    if (PinnedNodeArena* a = located_arena(key)) {
        ExpertKey ev = kNoEvictedKey;
        return a->reserve(key, &ev);
    }
    auto it = arenas_.find(node);
    if (it == arenas_.end()) return nullptr;
    PinnedNodeArena* a = it->second.get();
    if (!a->has_free_slot()) return nullptr;  // extend-only: never evict
    ExpertKey ev = kNoEvictedKey;
    void* s = a->reserve(key, &ev);
    if (s) {
        if (ev != kNoEvictedKey) {  // defensive (free slot)
            const int en = location_node(ev);
            location_.erase(ev);
            note_loc(ev, en, -1, 'E');
        }
        location_[key] = node;
        note_loc(key, -1, node, 'R');
    }
    return s;
}

size_t PinnedExpertArena::total_pinned_bytes() const {
    size_t total = 0;
    for (const auto& [node, arena] : arenas_) total += arena->total_bytes();
    return total;
}

size_t PinnedExpertArena::total_slots() const {
    size_t total = 0;
    for (const auto& [node, arena] : arenas_) total += arena->num_slots();
    return total;
}

std::vector<int> PinnedExpertArena::arena_nodes() const {
    std::vector<int> out;
    out.reserve(arenas_.size());
    for (const auto& [node, arena] : arenas_) out.push_back(node);
    std::sort(out.begin(), out.end());
    return out;
}

PinnedNodeArena* PinnedExpertArena::node_arena(int node) {
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

const PinnedNodeArena* PinnedExpertArena::node_arena(int node) const {
    auto it = arenas_.find(node);
    return it == arenas_.end() ? nullptr : it->second.get();
}

void* PinnedExpertArena::node_scratch(int node) const {
    const PinnedNodeArena* a = node_arena(node);
    return a ? a->node_scratch() : nullptr;
}

size_t PinnedExpertArena::node_scratch_bytes(int node) const {
    const PinnedNodeArena* a = node_arena(node);
    return a ? a->node_scratch_bytes() : 0;
}

size_t PinnedExpertArena::preload(
        const model::PrepackedSource& src, uint32_t num_layers,
        uint32_t num_experts, ArenaLoader* loader,
        const std::unordered_map<ExpertKey, int>* placement) {
    // Expert-MAJOR order (expert outer, layer inner): the decode working set
    // spans ALL layers, so spread coverage across layers rather than fill low
    // layers to capacity first.
    //
    // Placement goes through reserve_for_fill so preload obeys the SAME tiered
    // rules as the runtime miss path (INV-MoE-SPILL): the expert's home node
    // first (the preload-time "local" anchor — there is no consumer GPU yet),
    // then the weighted cross-node spill nodes (incl. a GPU-less node). This lets
    // the full weight set preload across all node arenas (home + spill) instead
    // of stopping at home-node capacity. The location index makes runtime resolve
    // find each expert wherever it landed (cross-NUMA H2D if non-local).

    // Map a NUMA node → any GPU attached to it (reserve_for_fill keys "local" off
    // the GPU's node). Built once. -1 entries (e.g. a GPU-less home node, which
    // cannot occur — home nodes are GPU-attached) fall back to home placement.
    std::unordered_map<int, int> gpu_on_node;
    for (int g = 0; g < numa_.num_gpus(); ++g) {
        int nd = numa_.gpu_numa_node(g);
        gpu_on_node.emplace(nd, g);   // first GPU per node
    }

    // Async path: route reads through the io_uring ArenaLoader (the SAME path the
    // runtime miss path uses — "imitate spill"), keeping the ring full so the
    // kernel overlaps reads across the NVMe queue. Worker-pool / no loader → keep
    // the simple synchronous pread (the prior behaviour).
    const bool async = loader &&
                       loader->backend() == ArenaLoader::Backend::kIoUring;

    size_t filled = 0, skipped_full = 0, failed = 0, outstanding = 0;
    size_t placed = 0, place_fallback = 0;  // arena host placement stats
    std::vector<ArenaLoadCompletion> comps;
    auto drain = [&](std::vector<ArenaLoadCompletion>& c) {
        for (const auto& comp : c) {
            if (comp.success) { mark_ready(comp.key); ++filled; } else { ++failed; }
            --outstanding;
        }
        c.clear();
    };

    size_t already = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        for (uint32_t L = 0; L < num_layers; ++L) {
            ExpertKey key{L, static_cast<uint16_t>(e)};
            if (!src.has(key)) continue;            // not a MoE layer / out of range
            // P-24b warm attach: slots adopted from the persisted arena are
            // already resident+ready — reloading them would be a wasted NVMe
            // read (and a pointless rewrite of identical bytes).
            if (is_ready(key)) { ++already; continue; }
            int home = numa_.expert_home_node(key.expert_idx);
            auto git = gpu_on_node.find(home);
            int gpu = (git != gpu_on_node.end()) ? git->second : -1;
            int node = -1;
            void* slot = nullptr;
            // Arena host placement (arena_placement.h): planned node first
            // (extend-only); fall back to the legacy tiered fill on a full
            // planned node or an unmapped key.
            if (placement) {
                auto pit = placement->find(key);
                if (pit != placement->end()) {
                    slot = reserve_on_node(key, pit->second);
                    if (slot) { node = pit->second; ++placed; }
                    else ++place_fallback;
                }
            }
            if (!slot)
                slot = reserve_for_fill(key, gpu, &node,
                                        /*extend_only=*/true);  // home → spill, no evict
            if (!slot) { ++skipped_full; continue; }  // every node arena full

            if (!async) {
                if (src.load_into(key, slot)) { mark_ready(key); ++filled; }
                else { ++failed; }                    // leaves slot reserved-empty
                continue;
            }

            // io_uring: submit; on a full ring (submit fails while reads are still
            // outstanding) block for ≥1 completion, then retry. A submit failure
            // with nothing outstanding is a genuine read-descriptor failure →
            // fall back to a synchronous pread for that slot.
            for (;;) {
                if (loader->submit(key, gpu, slot, &src)) { ++outstanding; break; }
                if (outstanding == 0) {
                    if (src.load_into(key, slot)) { mark_ready(key); ++filled; }
                    else { ++failed; }
                    break;
                }
                loader->wait_completed(comps);
                drain(comps);
            }
            // Opportunistic non-blocking reap to keep marking slots ready.
            loader->poll_completed(comps);
            drain(comps);
        }
    }
    // Drain the tail (async only; outstanding is always 0 in the sync path).
    while (outstanding > 0) {
        loader->wait_completed(comps);
        drain(comps);
    }

    spdlog::info("PinnedExpertArena: preloaded {} slots ({} adopted warm, "
                 "{} skipped — all arenas full, {} load failures){}",
                 filled, already, skipped_full, failed,
                 async ? " [io_uring]" : "");
    if (placement)
        spdlog::info("PinnedExpertArena: host placement ENGAGED — {} slots "
                     "placement-directed, {} fell back to tiered fill "
                     "(planned node full/absent)", placed, place_fallback);
    return filled;
}

}  // namespace layerstorm::memory
