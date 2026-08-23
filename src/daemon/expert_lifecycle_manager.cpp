// ExpertLifecycleManager — see spec/IPC_EXPERT_LIFECYCLE.md.

#include "daemon/expert_lifecycle_manager.h"
#include "daemon/state_transaction.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <spdlog/spdlog.h>

#include "core/memory/arena_loader.h"
#include "core/memory/pinned_expert_arena.h"
#include "core/perf_trace.h"
#include "model/weight_loader/weight_loader.h"
#include "model/weight_pipeline/packed_buffer_cache.h"
#include "model/weight_pipeline/prepacked_source.h"

namespace layerstorm::daemon {

static uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static uint32_t elapsed_us(uint64_t start_ns, uint64_t end_ns) {
    if (end_ns <= start_ns) return 0;
    return static_cast<uint32_t>((end_ns - start_ns) / 1000);
}

// ── Construction ────────────────────────────────────────────────────────────

ExpertLifecycleManager::ExpertLifecycleManager(Deps deps)
    : deps_(std::move(deps)),
      host_source_deps_{
          .pinned_arena     = deps_.pinned_arena,
          .arena_loader     = deps_.arena_loader,
          .prepacked_source = deps_.prepacked_source,
          .nvme_tier        = deps_.nvme_tier,
          .packed_cache     = deps_.packed_cache,
          .loaded_model     = deps_.loaded_model,
          .expert_shape     = deps_.expert_shape,
      },
      snapshot_(deps_.snapshot),
      first_moe_layer_(deps_.first_moe_layer),
      num_moe_layers_(deps_.num_moe_layers),
      num_experts_(deps_.num_experts),
      num_gpus_(deps_.num_gpus) {
    if (snapshot_) {
        snapshot_tx_ = std::make_unique<StateTransaction>(*snapshot_);
        bootstrap_snapshot();
    }
}

ExpertLifecycleManager::~ExpertLifecycleManager() = default;

// ── Host source resolution ──────────────────────────────────────────────────

ExpertLifecycleManager::HostSourceResult
ExpertLifecycleManager::resolve_host_source(memory::ExpertKey key,
                                            int target_gpu) {
    return transfer::resolve_expert_host_source(key, host_source_deps_,
                                                target_gpu);
}

// TEMP DEBUG (TD-100j deep x-ray): log the per-H2D micro-timing breakdown.
void ExpertLifecycleManager::log_h2d_micro() const {
    const auto& m = h2d_micro_;
    if (m.n == 0) { spdlog::info("[TD-100j x-ray] no H2D fetches"); return; }
    spdlog::info(
        "[TD-100j x-ray] H2D fetches={} cold(>5ms)={} ({:.1f}%) | "
        "resolve+load avg={:.3f} ms (max={:.3f}) | "
        "pre-DMA start->enqueue avg={:.3f} ms | "
        "DMA+detect enqueue->detected avg={:.3f} ms (max={:.3f})",
        m.n, m.cold, 100.0 * static_cast<double>(m.cold) / m.n,
        m.resolve_sum / 1e6 / m.n, m.resolve_max / 1e6,
        m.predma_sum / 1e6 / m.n,
        m.dma_sum / 1e6 / m.n, m.dma_max / 1e6);
}

void ExpertLifecycleManager::release_packed_host_buffer(memory::ExpertKey key) {
    // WP-3: prepacked source has process-lifetime mmap — nothing to release.
    if (deps_.prepacked_source && deps_.prepacked_source->has(key)) return;

    // WP-4: if cache is retaining (budget != 0), don't release — cache manages
    // lifecycle.  The caller has already reset host_buf_ref (shared_ptr copy),
    // so the cache's own reference keeps the buffer alive for reuse.
    if (deps_.packed_cache && !deps_.packed_cache->is_passthrough()
        && !deps_.packed_cache->is_mmap()) {
        return;
    }

    // Passthrough (budget=0) or no cache: release from cache + clear bundle.
    if (deps_.packed_cache) deps_.packed_cache->release(key);

    if (!deps_.loaded_model) return;
    auto li = static_cast<size_t>(key.layer_idx);
    auto ei = static_cast<size_t>(key.expert_idx);
    if (li >= deps_.loaded_model->layers.size()) return;
    auto& layer = deps_.loaded_model->layers[li];
    if (ei >= layer.routed_experts.size() || layer.routed_experts[ei].empty())
        return;
    auto& bundle = layer.routed_experts[ei][0];
    // Release lazily-packed buffer for both NVFP4 and FP8 (TD-93a).
    // Clear packed_slot only — weight.data (raw mmap projection) is preserved
    // so that re-pack on next prefetch can read valid data (TD-93h fix).
    bundle.owned_buf.reset();
    bundle.packed_slot = {};
}

HostTier ExpertLifecycleManager::compute_host_tier(
        memory::ExpertKey key) const {
    // WP-3: prepacked source — always warm if expert exists in prepacked files.
    if (deps_.prepacked_source && deps_.prepacked_source->has(key))
        return HostTier::kWarm;
    // Check mmap — always available if LoadedModel is present.
    if (deps_.loaded_model) {
        auto li = static_cast<size_t>(key.layer_idx);
        auto ei = static_cast<size_t>(key.expert_idx);
        if (li < deps_.loaded_model->layers.size()) {
            const auto& layer = deps_.loaded_model->layers[li];
            if (ei < layer.routed_experts.size() &&
                !layer.routed_experts[ei].empty()) {
                return HostTier::kWarm;  // mmap is always available
            }
        }
    }
    if (deps_.nvme_tier) {
        if (deps_.nvme_tier->is_in_host_ram(key)) return HostTier::kWarm;
        if (deps_.nvme_tier->is_inflight(key))    return HostTier::kLoadingToRam;
        if (deps_.nvme_tier->is_on_nvme(key))     return HostTier::kCold;
    }
    return HostTier::kCold;
}

// ── Snapshot publishing (ELM-8) ────────────────────────────────────────────

void ExpertLifecycleManager::publish_gpu_state(
        const GpuEntryKey& gk, GpuTier tier, uint32_t interest_count) {
    if (!snapshot_) return;
    uint32_t moe_layer = gk.key.layer_idx;
    if (moe_layer < first_moe_layer_) return;
    moe_layer -= first_moe_layer_;
    if (moe_layer >= ipc::kMaxMoeLayers ||
        gk.key.expert_idx >= ipc::kMaxExperts ||
        gk.gpu_idx < 0 || gk.gpu_idx >= static_cast<int>(ipc::kMaxGpus)) return;

    StateTransactionGuard guard(*snapshot_tx_);
    size_t flat = static_cast<size_t>(moe_layer) * ipc::kMaxExperts * ipc::kMaxGpus
                + static_cast<size_t>(gk.key.expert_idx) * ipc::kMaxGpus
                + static_cast<size_t>(gk.gpu_idx);
    snapshot_->expert_gpu_tier[flat] = static_cast<uint8_t>(tier);
    snapshot_->expert_interest_count[flat] = static_cast<uint8_t>(
        std::min(interest_count, uint32_t(255)));

    // Residency bitmap: set if fully ready (tier >= kHot), clear otherwise.
    if (tier >= GpuTier::kHot) {
        snapshot_->residency_bitmap[flat / 8] |=
            static_cast<uint8_t>(1u << (flat % 8));
    } else {
        snapshot_->residency_bitmap[flat / 8] &=
            static_cast<uint8_t>(~(1u << (flat % 8)));
    }

    size_t expert_flat = static_cast<size_t>(moe_layer) * ipc::kMaxExperts
                       + gk.key.expert_idx;
    snapshot_->expert_last_change_ns[expert_flat] = now_ns();
}

void ExpertLifecycleManager::publish_host_state(
        memory::ExpertKey key, HostTier tier) {
    if (!snapshot_) return;
    uint32_t moe_layer = key.layer_idx;
    if (moe_layer < first_moe_layer_) return;
    moe_layer -= first_moe_layer_;
    if (moe_layer >= ipc::kMaxMoeLayers ||
        key.expert_idx >= ipc::kMaxExperts) return;

    StateTransactionGuard guard(*snapshot_tx_);
    size_t idx = static_cast<size_t>(moe_layer) * ipc::kMaxExperts + key.expert_idx;
    snapshot_->host_tier[idx] = static_cast<uint8_t>(tier);

    // Update host_resident_bitmap bit.
    if (tier == HostTier::kWarm) {
        snapshot_->host_resident_bitmap[idx / 8] |=
            static_cast<uint8_t>(1u << (idx % 8));
    } else {
        snapshot_->host_resident_bitmap[idx / 8] &=
            static_cast<uint8_t>(~(1u << (idx % 8)));
    }

    // Update NUMA node: query NvmeTier when warm, -1 otherwise.
    int8_t numa = -1;
    if (tier == HostTier::kWarm && deps_.nvme_tier) {
        numa = static_cast<int8_t>(deps_.nvme_tier->host_numa_node(key));
    } else if (tier == HostTier::kWarm && deps_.loaded_model) {
        numa = 0;  // mmap — assume local node
    }
    snapshot_->expert_host_numa[idx] = numa;

    // Per-NUMA tier (ELM-8b).
    size_t numa_base = idx * ipc::kMaxNuma;
    if (tier == HostTier::kWarm && numa >= 0 &&
        numa < static_cast<int8_t>(ipc::kMaxNuma)) {
        for (int n = 0; n < static_cast<int>(ipc::kMaxNuma); ++n) {
            snapshot_->host_numa_tier[numa_base + n] =
                (n == numa) ? static_cast<uint8_t>(HostTier::kWarm)
                            : static_cast<uint8_t>(HostTier::kCold);
        }
    } else {
        std::memset(&snapshot_->host_numa_tier[numa_base], 0, ipc::kMaxNuma);
    }

    snapshot_->expert_last_change_ns[idx] = now_ns();
}

void ExpertLifecycleManager::bootstrap_snapshot() {
    if (!snapshot_ || !deps_.expert_cache) return;

    StateTransactionGuard guard(*snapshot_tx_);
    auto ts = now_ns();

    // Zero the active region of all ELM-published arrays.
    size_t active_triples = static_cast<size_t>(num_moe_layers_) *
                            ipc::kMaxExperts * ipc::kMaxGpus;
    size_t active_experts = static_cast<size_t>(num_moe_layers_) * ipc::kMaxExperts;

    if (active_triples > 0) {
        std::memset(snapshot_->expert_gpu_tier, 0, active_triples);
        std::memset(snapshot_->expert_interest_count, 0, active_triples);
        std::memset(snapshot_->residency_bitmap, 0, active_triples / 8);
    }
    if (active_experts > 0) {
        std::memset(snapshot_->host_resident_bitmap, 0, active_experts / 8);
        std::memset(snapshot_->host_tier, 0, active_experts);
        std::memset(snapshot_->expert_host_numa, 0xFF, active_experts);  // -1
        std::memset(snapshot_->host_numa_tier, 0,
                    active_experts * ipc::kMaxNuma);  // all kCold
        std::memset(snapshot_->expert_last_change_ns, 0,
                    active_experts * sizeof(uint64_t));
    }

    // Populate from ExpertCache for pre-existing entries.
    for (int g = 0; g < num_gpus_ && g < static_cast<int>(ipc::kMaxGpus); ++g) {
        auto entries = deps_.expert_cache->residency_snapshot(g);
        for (const auto& ri : entries) {
            if (ri.key.layer_idx < first_moe_layer_) continue;
            uint32_t moe_layer = ri.key.layer_idx - first_moe_layer_;
            if (moe_layer >= ipc::kMaxMoeLayers) continue;
            if (ri.key.expert_idx >= ipc::kMaxExperts) continue;
            if (ri.gpu_idx < 0 || ri.gpu_idx >= static_cast<int>(ipc::kMaxGpus)) continue;

            // GPU tier.
            GpuTier gt = GpuTier::kReserved;
            if (ri.sub_components_ready ==
                static_cast<uint8_t>(memory::SubComponent::kAll)) {
                gt = GpuTier::kHot;
            }
            size_t flat = static_cast<size_t>(moe_layer) * ipc::kMaxExperts * ipc::kMaxGpus
                        + static_cast<size_t>(ri.key.expert_idx) * ipc::kMaxGpus
                        + static_cast<size_t>(ri.gpu_idx);
            snapshot_->expert_gpu_tier[flat] = static_cast<uint8_t>(gt);
            if (gt == GpuTier::kHot) {
                snapshot_->residency_bitmap[flat / 8] |=
                    static_cast<uint8_t>(1u << (flat % 8));
            }

            // Also populate internal state for ELM tracking.
            GpuEntryKey gk{ri.key, ri.gpu_idx};
            auto& ge = gpu_entries_[gk];
            ge.tier = gt;

            // Host tier.
            size_t expert_idx = static_cast<size_t>(moe_layer) * ipc::kMaxExperts
                              + ri.key.expert_idx;
            HostTier ht = compute_host_tier(ri.key);
            snapshot_->host_tier[expert_idx] = static_cast<uint8_t>(ht);
            if (ht == HostTier::kWarm) {
                snapshot_->host_resident_bitmap[expert_idx / 8] |=
                    static_cast<uint8_t>(1u << (expert_idx % 8));
                // NUMA node from NvmeTier or mmap.
                int8_t numa = 0;  // default: mmap on local node
                if (deps_.nvme_tier)
                    numa = static_cast<int8_t>(deps_.nvme_tier->host_numa_node(ri.key));
                snapshot_->expert_host_numa[expert_idx] = numa;
                // Per-NUMA tier (ELM-8b).
                if (numa >= 0 && numa < static_cast<int8_t>(ipc::kMaxNuma))
                    snapshot_->host_numa_tier[expert_idx * ipc::kMaxNuma + numa] =
                        static_cast<uint8_t>(HostTier::kWarm);
            }
            snapshot_->expert_last_change_ns[expert_idx] = ts;

            // Populate host entry.
            auto& he = host_entries_[ri.key];
            if (he.tier == HostTier::kCold) {
                he.tier = ht;
            }
        }
    }

    // Also populate host tier for experts that are warm but not on any GPU
    // (mmap-available or prepacked-source experts).
    if (deps_.loaded_model || deps_.prepacked_source) {
        for (uint32_t l = 0; l < num_moe_layers_; ++l) {
            uint32_t abs_layer = l + first_moe_layer_;
            for (uint32_t e = 0; e < num_experts_; ++e) {
                memory::ExpertKey key{abs_layer, static_cast<uint16_t>(e)};
                size_t idx = static_cast<size_t>(l) * ipc::kMaxExperts + e;
                if (snapshot_->host_tier[idx] != 0) continue;  // already set
                HostTier ht = compute_host_tier(key);
                snapshot_->host_tier[idx] = static_cast<uint8_t>(ht);
                if (ht == HostTier::kWarm) {
                    snapshot_->host_resident_bitmap[idx / 8] |=
                        static_cast<uint8_t>(1u << (idx % 8));
                    snapshot_->expert_host_numa[idx] = 0;  // mmap on local node
                    // Per-NUMA tier: mmap on NUMA 0 (ELM-8b).
                    snapshot_->host_numa_tier[idx * ipc::kMaxNuma + 0] =
                        static_cast<uint8_t>(HostTier::kWarm);
                    snapshot_->expert_last_change_ns[idx] = ts;
                }
            }
        }
    }
}

// ── ensure_resident ─────────────────────────────────────────────────────────

LifecycleToken ExpertLifecycleManager::ensure_resident(
        memory::ExpertKey key, int gpu, memory::CacheZone zone,
        uint32_t cmd_seq, float priority, int delay_us) {
    auto token = next_token_++;
    GpuEntryKey gk{key, gpu};

    auto& ge = gpu_entries_[gk];
    auto& he = host_entries_[key];

    // Initialize host tier if first access.
    if (he.tier == HostTier::kCold && he.gpu_waiters.empty() && he.nvme_token == 0) {
        he.tier = compute_host_tier(key);
    }

    Interest interest{token, cmd_seq, zone};
    token_index_[token] = gk;
    cmd_seq_to_token_.emplace(cmd_seq, token);

    switch (ge.tier) {
        case GpuTier::kHot:
        case GpuTier::kPartial1:
        case GpuTier::kPartial1_2: {
            // Already resident (or partially) — immediate completion.
            pending_completions_.push_back({token, cmd_seq, key, gpu, true});
            return token;
        }

        case GpuTier::kTransferring: {
            // H2D already inflight — just add interest. If the new interest is
            // more urgent (a demand fetch joining a transfer enqueued at
            // speculative-prefetch priority), re-assert its priority in the
            // transfer engine's staged admission queue — a demand fetch must
            // never ride a prefetch's low priority (no-op once dispatched).
            ge.interests.push_back(interest);
            if (priority > ge.priority) {
                ge.priority = priority;
                if (ge.transfer_token != 0 && deps_.transfer_engine)
                    deps_.transfer_engine->reprioritize(ge.transfer_token,
                                                        priority);
            }
            publish_gpu_state(gk, ge.tier,
                              static_cast<uint32_t>(ge.interests.size()));
            return token;
        }

        case GpuTier::kReserved: {
            // Slot allocated but no transfer yet.  Might be waiting for host.
            // Re-assert priority for the pending try_start_h2d (it enqueues at
            // ge.priority): a demand interest joining a low-priority prefetch
            // reservation must lift the eventual H2D to demand priority.
            if (priority > ge.priority) ge.priority = priority;
            ge.interests.push_back(interest);
            publish_gpu_state(gk, ge.tier,
                              static_cast<uint32_t>(ge.interests.size()));
            if (he.tier == HostTier::kWarm) {
                try_start_h2d(gk, ge, he);
            }
            // If host not warm, we're already queued as a gpu_waiter (from
            // the original ensure_resident that created the RESERVED state).
            return token;
        }

        case GpuTier::kAbsent:
            break;  // Fall through to start chain.

        case GpuTier::kDraining:
            // D2H in progress before eviction (slow evict path).
            // Cannot start a new ensure_resident while draining.
            pending_completions_.push_back({token, cmd_seq, key, gpu, false});
            return token;
    }

    // GPU tier is ABSENT.  Start the chain based on host tier.
    ge.interests.push_back(interest);
    ge.start_ns = now_ns();
    ge.priority = priority;
    ge.delay_us = delay_us;

    switch (he.tier) {
        case HostTier::kWarm: {
            // Host data available — reserve VRAM slot and start H2D.
            void* addr = deps_.expert_cache->reserve(key, gpu, zone, false);
            if (!addr) {
                // VRAM full.
                complete_interests(gk, ge, false, pending_completions_);
                return token;
            }
            ge.tier = GpuTier::kReserved;
            publish_gpu_state(gk, ge.tier,
                              static_cast<uint32_t>(ge.interests.size()));
            try_start_h2d(gk, ge, he);
            return token;
        }

        case HostTier::kLoadingToRam: {
            // NVMe read already inflight — queue as GPU waiter.
            he.gpu_waiters.push_back({gpu, zone});
            return token;
        }

        case HostTier::kCold: {
            // Need NVMe read first.
            if (deps_.nvme_tier) {
                auto nvme_tok = deps_.nvme_tier->read_expert(key, gpu);
                if (nvme_tok.has_value()) {
                    if (nvme_tok.value() == 0) {
                        // WP-5: immediate completion (mmap path) — now warm.
                        he.tier = HostTier::kWarm;
                        publish_host_state(key, he.tier);
                        // Fall through to start H2D directly.
                        void* addr = deps_.expert_cache->reserve(
                            key, gpu, zone, false);
                        if (!addr) {
                            complete_interests(gk, ge, false,
                                               pending_completions_);
                            return token;
                        }
                        ge.tier = GpuTier::kReserved;
                        publish_gpu_state(gk, ge.tier,
                                          static_cast<uint32_t>(
                                              ge.interests.size()));
                        try_start_h2d(gk, ge, he);
                        return token;
                    }
                    // Async token (future use).
                    he.tier = HostTier::kLoadingToRam;
                    publish_host_state(key, he.tier);
                    he.nvme_token = nvme_tok.value();
                    he.gpu_waiters.push_back({gpu, zone});
                    return token;
                }
            }
            // NVMe read failed or no NvmeTier — cannot load expert.
            complete_interests(gk, ge, false, pending_completions_);
            return token;
        }
    }

    return token;
}

// ── ensure_residents (batched, inline make-room evict) ───────────────────────

int ExpertLifecycleManager::ensure_residents(
        int gpu, std::span<const ResidentReq> reqs,
        std::vector<memory::ExpertKey>& victims,
        uint32_t cmd_seq, float priority) {
    size_t vc = 0;  // victim cursor (cheapest-first, caller pre-filtered)
    int progressed = 0;
    for (const auto& req : reqs) {
        // ABSENT + full stable zone → free exactly one slot inline so the reserve
        // inside ensure_resident succeeds instead of returning null, dropping the
        // interest, and parking ~30 ms (TD-FAR-SLOT-RESERVE-STALL). Stable zone
        // only — the evict board that produced `victims` tracks stable residency.
        // request_evict skips a victim still un-evictable (kTransferring / pending
        // interests / kDraining), so advance the cursor until one frees a slot.
        if (req.zone == memory::CacheZone::kStable
            && state(req.key, gpu).gpu_tier == GpuTier::kAbsent
            && deps_.expert_cache->free_slots(gpu, memory::CacheZone::kStable) <= 0) {
            while (vc < victims.size()) {
                if (request_evict(victims[vc++], gpu)) break;
            }
        }
        ensure_resident(req.key, gpu, req.zone, cmd_seq, priority, 0);
        if (state(req.key, gpu).gpu_tier != GpuTier::kAbsent) ++progressed;
    }
    return progressed;
}

// ── try_start_h2d ───────────────────────────────────────────────────────────

bool ExpertLifecycleManager::try_start_h2d(
        const GpuEntryKey& gk, GpuEntry& ge, HostEntry& /*he*/) {
    const uint32_t pt_k = (gk.key.layer_idx << 16) | (gk.key.expert_idx & 0xffffu);
    perf_trace::record(perf_trace::kEnsureEnter,
                       static_cast<uint16_t>(gk.gpu_idx), 0, pt_k, 0);
    // Consumer-aware placement (INV-MoE-NUMA): source the host copy from the
    // computing GPU's local-node arena so the H2D is local-bandwidth.
    const uint64_t t_resolve0 = now_ns();  // TEMP DEBUG (TD-100j)
    auto hs = resolve_host_source(gk.key, gk.gpu_idx);
    ge.resolve_ns = now_ns() - t_resolve0;  // TEMP DEBUG (TD-100j)
    perf_trace::record(perf_trace::kResolveExit,
                       static_cast<uint16_t>(gk.gpu_idx), 0, pt_k, 0);

    // J-1: arena cold load is in flight asynchronously — defer the H2D. The GPU
    // entry stays RESERVED; record it as a waiter so process_arena_loads()
    // re-drives this once the slot is filled. The daemon keeps cycling
    // (INV-3.4.2) instead of blocking on the ~150 ms file→slot copy.
    if (hs.status == transfer::HostSourceStatus::kPending) {
        auto& waiters = arena_load_waiters_[gk.key];
        if (std::find(waiters.begin(), waiters.end(), gk) == waiters.end())
            waiters.push_back(gk);
        return false;
    }

    // Stage 2 direct/mmap-free: arena momentarily full (all slots pinned mid-H2D)
    // — no load submitted, so re-drive when a slot frees (H2D completion), not on
    // an arena-load completion. Park in slot_pressure_waiters_; the entry stays
    // RESERVED. (Only reachable when the arena is smaller than the in-flight set.)
    if (hs.status == transfer::HostSourceStatus::kBackpressure) {
        if (std::find(slot_pressure_waiters_.begin(),
                      slot_pressure_waiters_.end(), gk) ==
            slot_pressure_waiters_.end())
            slot_pressure_waiters_.push_back(gk);
        return false;
    }

    const void* src = hs.ptr;
    auto host_buf = hs.owned_buf;
    bool pinned = hs.pinned;
    if (!src) return false;

    const auto* entry = deps_.expert_cache->lookup(gk.key, gk.gpu_idx);
    if (!entry || !entry->vram_address) return false;

    // GG-9 (TD-GG9-EXPERT-PERLAYER-COPYSIZE): the VRAM cache slot is sized at the
    // global per-projection MAX k-quant (expert_bytes()), but a mixed "XL" GGUF
    // packs each layer's experts at its OWN (possibly smaller) per-projection
    // types — so the host source buffer can be SMALLER than the slot. Copy the
    // actual source size when known; copying the full slot would read past the
    // end of the smaller host buffer (segfault). The over-sized slot tail stays
    // unused — the expert GEMM reads only this layer's per-type offsets.
    auto bytes = deps_.expert_cache->expert_bytes();
    if (hs.bytes > 0 && hs.bytes < bytes) bytes = hs.bytes;
    auto token = deps_.transfer_engine->enqueue_h2d(
        gk.key, gk.gpu_idx, entry->vram_address, src, bytes,
        ge.priority, ge.delay_us,
        {}, /*needs_pinned_staging=*/!pinned);

    if (token.has_value()) {
        if (pinned) ++h2d_path_stats_.direct;   // 481-2: full-bandwidth source
        else        ++h2d_path_stats_.staged;   // unpinned → staging copy
        // M3b: fresh enqueue = one demand fetch on the wire (online-placement
        // frequency signal; the dedup branch below intentionally not counted;
        // BIG/prefill-class waves gated off via set_fetch_count_enabled).
        if (h2d_enqueue_observer_ && fetch_count_enabled_)
            h2d_enqueue_observer_(gk.key);
        ge.tier = GpuTier::kTransferring;
        ge.transfer_token = token.value();
        ge.h2d_start_ns = now_ns();
        ge.host_buf_ref = std::move(host_buf);  // TD-82a: keep alive during DMA
        publish_gpu_state(gk, ge.tier,
                          static_cast<uint32_t>(ge.interests.size()));
        if (deps_.emit_progress && !ge.interests.empty()) {
            pending_progress_.push_back(
                {gk.key, gk.gpu_idx, 3, ge.interests.front().cmd_seq});
        }
        return true;
    }

    // Dedup — transfer already inflight (possibly from an atomic H2D command).
    // Treat as TRANSFERRING — poll() will pick up the completion.
    if (deps_.transfer_engine->is_inflight_h2d(gk.key, gk.gpu_idx)) {
        ge.tier = GpuTier::kTransferring;
        ge.transfer_token = 0;  // unknown token — match by key in poll()
        ge.host_buf_ref = std::move(host_buf);  // TD-82a
        publish_gpu_state(gk, ge.tier,
                          static_cast<uint32_t>(ge.interests.size()));
        return true;
    }

    return false;
}

// ── process_arena_loads (J-1) ────────────────────────────────────────────────

void ExpertLifecycleManager::process_arena_loads(
        std::vector<LifecycleCompletion>& out) {
    if (!deps_.arena_loader || !deps_.pinned_arena) return;

    std::vector<memory::ArenaLoadCompletion> loads;
    deps_.arena_loader->poll_completed(loads);
    if (loads.empty()) return;

    for (const auto& lc : loads) {
        // Mark the slot ready (success) or clear its loading flag so it can be
        // re-loaded/evicted (failure). Use the consumer-local arena (the same
        // slot resolve()/reserve() returned for this gpu — INV-MoE-NUMA).
        if (lc.success) {
            deps_.pinned_arena->mark_ready(lc.key, lc.gpu);
        } else {
            deps_.pinned_arena->clear_loading(lc.key, lc.gpu);
        }

        auto wit = arena_load_waiters_.find(lc.key);
        if (wit == arena_load_waiters_.end()) continue;
        // Copy + erase before re-driving: try_start_h2d may re-insert into
        // arena_load_waiters_ (e.g. if a NEW load is somehow needed), and we
        // must not iterate a container we mutate.
        auto waiters = std::move(wit->second);
        arena_load_waiters_.erase(wit);

        for (const auto& gk : waiters) {
            auto git = gpu_entries_.find(gk);
            if (git == gpu_entries_.end()) continue;
            auto& ge = git->second;
            // The entry must still be RESERVED and waiting (it may have been
            // cancelled/reverted, or already advanced via another path).
            if (ge.tier != GpuTier::kReserved) continue;

            if (lc.success) {
                auto& he = host_entries_[gk.key];
                try_start_h2d(gk, ge, he);  // slot now ready → enqueues H2D
            } else {
                // Cold load failed — fail the interests for this entry.
                complete_interests(gk, ge, false, out);
            }
        }
    }
}

// ── request_evict ───────────────────────────────────────────────────────────

bool ExpertLifecycleManager::request_evict(memory::ExpertKey key, int gpu) {
    GpuEntryKey gk{key, gpu};
    auto it = gpu_entries_.find(gk);

    if (it == gpu_entries_.end()) {
        // Not tracked by ELM — delegate directly to ExpertCache.
        return deps_.expert_cache->evict(key, gpu);
    }

    auto& ge = it->second;

    // INV-ELM-2: refuse eviction while interests pending.
    if (!ge.interests.empty()) return false;

    if (ge.tier == GpuTier::kTransferring || ge.tier == GpuTier::kDraining)
        return false;

    bool ok = deps_.expert_cache->evict(key, gpu);
    if (ok) {
        ge.tier = GpuTier::kAbsent;
        ge.transfer_token = 0;
        ge.host_buf_ref.reset();  // Defensive: release packed buffer ref (TD-82j).
        publish_gpu_state(gk, ge.tier, 0);
    }
    return ok;
}

// ── request_drain (ELM-6) ──────────────────────────────────────────────────

bool ExpertLifecycleManager::request_drain(
        memory::ExpertKey key, int gpu, uint32_t cmd_seq) {
    GpuEntryKey gk{key, gpu};
    auto it = gpu_entries_.find(gk);

    if (it == gpu_entries_.end()) return false;
    auto& ge = it->second;
    if (ge.tier != GpuTier::kHot) return false;

    // INV-ELM-2: refuse while interests pending.
    if (!ge.interests.empty()) return false;

    if (!deps_.numa_manager || !deps_.transfer_engine || !deps_.nvme_tier)
        return false;

    const auto* entry = deps_.expert_cache->lookup(key, gpu);
    if (!entry || !entry->vram_address) return false;

    auto bytes = deps_.expert_cache->expert_bytes();
    auto host_buf = deps_.numa_manager->allocate_for_gpu(
        static_cast<size_t>(bytes), gpu);

    auto token = deps_.transfer_engine->enqueue_d2h(
        key, gpu, host_buf.data, entry->vram_address, bytes);

    if (!token.has_value()) {
        deps_.numa_manager->free(host_buf);
        return false;
    }

    ge.tier = GpuTier::kDraining;
    ge.transfer_token = token.value();
    ge.drain_cmd_seq = cmd_seq;
    ge.drain_host_buf = host_buf;
    ge.drain_start_ns = now_ns();

    publish_gpu_state(gk, ge.tier, 0);
    return true;
}

// ── cancel ──────────────────────────────────────────────────────────────────

void ExpertLifecycleManager::cancel(LifecycleToken token) {
    auto tit = token_index_.find(token);
    if (tit == token_index_.end()) return;

    GpuEntryKey gk = tit->second;
    token_index_.erase(tit);

    auto git = gpu_entries_.find(gk);
    if (git == gpu_entries_.end()) return;

    auto& ge = git->second;

    // Remove this interest and its cmd_seq mapping.
    auto& interests = ge.interests;
    for (auto it = interests.begin(); it != interests.end(); ++it) {
        if (it->token == token) {
            // Erase only this specific (cmd_seq, token) pair from the multimap.
            auto [begin, end] = cmd_seq_to_token_.equal_range(it->cmd_seq);
            for (auto mit = begin; mit != end; ++mit) {
                if (mit->second == token) {
                    cmd_seq_to_token_.erase(mit);
                    break;
                }
            }
            interests.erase(it);
            break;
        }
    }

    // Also remove from pending_completions_ (if it was an immediate completion).
    pending_completions_.erase(
        std::remove_if(pending_completions_.begin(), pending_completions_.end(),
                        [token](const LifecycleCompletion& c) { return c.token == token; }),
        pending_completions_.end());

    // If refcount dropped to 0, cancel underlying operations.
    // revert_gpu_entry publishes kAbsent internally — only publish here
    // for the partial-cancel case (interests still pending).
    if (interests.empty()) {
        revert_gpu_entry(gk, ge);
    } else {
        publish_gpu_state(gk, ge.tier,
                          static_cast<uint32_t>(ge.interests.size()));
    }
}

// ── revert_gpu_entry ────────────────────────────────────────────────────────

void ExpertLifecycleManager::revert_gpu_entry(
        const GpuEntryKey& gk, GpuEntry& ge) {
    switch (ge.tier) {
        case GpuTier::kTransferring:
            if (ge.transfer_token != 0) {
                deps_.transfer_engine->cancel(ge.transfer_token);
            }
            deps_.expert_cache->evict(gk.key, gk.gpu_idx);
            break;
        case GpuTier::kReserved:
            deps_.expert_cache->evict(gk.key, gk.gpu_idx);
            break;
        case GpuTier::kDraining:
            if (ge.transfer_token != 0) {
                deps_.transfer_engine->cancel(ge.transfer_token);
            }
            if (ge.drain_host_buf.data && deps_.numa_manager) {
                deps_.numa_manager->free(ge.drain_host_buf);
            }
            ge.drain_host_buf = {};
            ge.drain_cmd_seq = 0;
            ge.drain_start_ns = 0;
            deps_.expert_cache->evict(gk.key, gk.gpu_idx);
            break;
        case GpuTier::kHot:
        case GpuTier::kPartial1:
        case GpuTier::kPartial1_2:
        case GpuTier::kAbsent:
            break;
    }
    ge.tier = GpuTier::kAbsent;
    ge.transfer_token = 0;
    ge.host_buf_ref.reset();  // TD-82a
    publish_gpu_state(gk, ge.tier, 0);

    // Also remove any GPU waiter from host entry.
    auto hit = host_entries_.find(gk.key);
    if (hit != host_entries_.end()) {
        auto& waiters = hit->second.gpu_waiters;
        waiters.erase(
            std::remove_if(waiters.begin(), waiters.end(),
                            [&](const PendingGpuChain& w) { return w.gpu_idx == gk.gpu_idx; }),
            waiters.end());
    }
}

// ── complete_interests ──────────────────────────────────────────────────────

void ExpertLifecycleManager::complete_interests(
        const GpuEntryKey& gk, GpuEntry& ge, bool success,
        std::vector<LifecycleCompletion>& out) {
    auto end_ns = now_ns();
    uint32_t nvme_us  = elapsed_us(ge.start_ns, ge.nvme_done_ns);
    uint32_t dma_us   = elapsed_us(ge.h2d_start_ns, end_ns);
    uint32_t total_us = elapsed_us(ge.start_ns, end_ns);

    // TEMP DEBUG (TD-100j deep x-ray): accumulate H2D micro-segments.
    if (success && ge.h2d_start_ns != 0) {
        ++h2d_micro_.n;
        h2d_micro_.resolve_sum += ge.resolve_ns;
        h2d_micro_.resolve_max = std::max(h2d_micro_.resolve_max, ge.resolve_ns);
        if (ge.resolve_ns > 5'000'000ULL) ++h2d_micro_.cold;   // >5 ms = cold load
        if (ge.h2d_start_ns > ge.start_ns)
            h2d_micro_.predma_sum += ge.h2d_start_ns - ge.start_ns;
        const uint64_t dma_ns = (end_ns > ge.h2d_start_ns)
                              ? end_ns - ge.h2d_start_ns : 0;
        h2d_micro_.dma_sum += dma_ns;
        h2d_micro_.dma_max = std::max(h2d_micro_.dma_max, dma_ns);
    }

    const uint32_t pt_k = (gk.key.layer_idx << 16) | (gk.key.expert_idx & 0xffffu);
    for (const auto& interest : ge.interests) {
        out.push_back({interest.token, interest.cmd_seq, gk.key, gk.gpu_idx,
                       success, /*is_eviction=*/false, nvme_us, dma_us, total_us});
        perf_trace::record(perf_trace::kReadyPosted,
                           static_cast<uint16_t>(gk.gpu_idx), interest.cmd_seq,
                           pt_k, 0);
        token_index_.erase(interest.token);
        // Erase only this specific (cmd_seq, token) pair from the multimap.
        auto [begin, end] = cmd_seq_to_token_.equal_range(interest.cmd_seq);
        for (auto mit = begin; mit != end; ++mit) {
            if (mit->second == interest.token) {
                cmd_seq_to_token_.erase(mit);
                break;
            }
        }
    }
    ge.interests.clear();

    if (!success) {
        // Revert on failure.
        revert_gpu_entry(gk, ge);
    }
}

// ── state queries ───────────────────────────────────────────────────────────

ExpertState ExpertLifecycleManager::state(
        memory::ExpertKey key, int gpu) const {
    HostTier ht = host_state(key);

    GpuTier gt = GpuTier::kAbsent;
    uint32_t ic = 0;
    GpuEntryKey gk{key, gpu};
    auto it = gpu_entries_.find(gk);
    if (it != gpu_entries_.end()) {
        gt = it->second.tier;
        ic = static_cast<uint32_t>(it->second.interests.size());
    } else if (deps_.expert_cache->is_resident(key, gpu)) {
        // Managed outside ELM (atomic commands).
        const auto* entry = deps_.expert_cache->lookup(key, gpu);
        if (entry && entry->sub_components_ready == static_cast<uint8_t>(memory::SubComponent::kAll)) {
            gt = GpuTier::kHot;
        } else if (entry) {
            gt = GpuTier::kReserved;
        }
    }

    return {ht, gt, ic};
}

HostTier ExpertLifecycleManager::host_state(memory::ExpertKey key) const {
    auto it = host_entries_.find(key);
    if (it != host_entries_.end()) return it->second.tier;
    return compute_host_tier(key);
}

// ── find_by_cmd_seq ────────────────────────────────────────────────────────

LifecycleToken ExpertLifecycleManager::find_by_cmd_seq(uint32_t cmd_seq) const {
    auto it = cmd_seq_to_token_.find(cmd_seq);
    return it != cmd_seq_to_token_.end() ? it->second : 0;
}

std::vector<LifecycleToken> ExpertLifecycleManager::find_all_by_cmd_seq(
        uint32_t cmd_seq) const {
    std::vector<LifecycleToken> tokens;
    auto [begin, end] = cmd_seq_to_token_.equal_range(cmd_seq);
    for (auto it = begin; it != end; ++it) {
        tokens.push_back(it->second);
    }
    return tokens;
}

// ── poll ────────────────────────────────────────────────────────────────────

PollResult ExpertLifecycleManager::poll(
        std::span<const transfer::TransferCompletion> transfer_completions,
        std::span<const memory::IoCompletion> nvme_completions) {
    PollResult result;

    // Collect pending immediate completions.
    result.lifecycle.insert(result.lifecycle.end(),
                            pending_completions_.begin(),
                            pending_completions_.end());
    pending_completions_.clear();

    result.progress = std::move(pending_progress_);
    pending_progress_.clear();

    // J-1: reap completed async arena cold loads BEFORE NVMe/transfer so a slot
    // that became ready this cycle can enqueue its H2D immediately. Keeps the
    // poll loop non-blocking (INV-3.4.2): the ~150 ms file→slot copy ran on the
    // ArenaLoader worker pool, not here.
    process_arena_loads(result.lifecycle);

    // 1. Process NVMe completions — advance LOADING_TO_RAM → WARM.
    for (const auto& nc : nvme_completions) {
        auto hit = host_entries_.find(nc.key);
        if (hit == host_entries_.end() ||
            hit->second.tier != HostTier::kLoadingToRam) {
            result.unhandled_nvme.push_back(nc);
            continue;
        }
        auto& he = hit->second;

        if (nc.success) {
            he.tier = HostTier::kWarm;
            he.nvme_token = 0;
            publish_host_state(nc.key, he.tier);
            auto nvme_done = now_ns();

            // Chain: start GPU-tier operations for all waiters.
            for (const auto& waiter : he.gpu_waiters) {
                GpuEntryKey gk{nc.key, waiter.gpu_idx};
                auto& ge = gpu_entries_[gk];
                ge.nvme_done_ns = nvme_done;

                if (ge.tier == GpuTier::kAbsent) {
                    void* addr = deps_.expert_cache->reserve(
                        nc.key, waiter.gpu_idx, waiter.zone, false);
                    if (addr) {
                        ge.tier = GpuTier::kReserved;
                        publish_gpu_state(gk, ge.tier,
                                          static_cast<uint32_t>(ge.interests.size()));
                        try_start_h2d(gk, ge, he);
                    } else {
                        complete_interests(gk, ge, false, result.lifecycle);
                    }
                } else if (ge.tier == GpuTier::kReserved) {
                    try_start_h2d(gk, ge, he);
                }
            }
            he.gpu_waiters.clear();
        } else {
            he.tier = HostTier::kCold;
            he.nvme_token = 0;
            publish_host_state(nc.key, he.tier);
            for (const auto& waiter : he.gpu_waiters) {
                GpuEntryKey gk{nc.key, waiter.gpu_idx};
                auto git = gpu_entries_.find(gk);
                if (git != gpu_entries_.end()) {
                    complete_interests(gk, git->second, false, result.lifecycle);
                }
            }
            he.gpu_waiters.clear();
        }
    }

    // 2. Process transfer completions — advance TRANSFERRING → HOT (H2D)
    //    and DRAINING → ABSENT (D2H, ELM-6).
    for (const auto& xc : transfer_completions) {
        GpuEntryKey gk{xc.key, xc.gpu_idx};
        auto git = gpu_entries_.find(gk);

        if (xc.direction == transfer::TransferDirection::kH2D) {
            if (git == gpu_entries_.end() ||
                git->second.tier != GpuTier::kTransferring) {
                result.unhandled_transfers.push_back(xc);
                continue;
            }
            auto& ge = git->second;

            if (xc.success) {
                deps_.expert_cache->mark_all_ready(gk.key, gk.gpu_idx);
                ge.tier = GpuTier::kHot;
                ge.transfer_token = 0;
                ge.host_buf_ref.reset();           // TD-82a: DMA done, release ref
                release_packed_host_buffer(gk.key); // TD-82a: release LoadedModel copy
                publish_gpu_state(gk, ge.tier,
                                  static_cast<uint32_t>(ge.interests.size()));
                complete_interests(gk, ge, true, result.lifecycle);
                // After complete_interests clears interests, publish count=0.
                publish_gpu_state(gk, ge.tier, 0);
                // Stage 2: this H2D freed an arena slot — re-drive any entries
                // deferred under arena back-pressure (direct mode, all-slots-
                // in-flight). Copy+clear first (try_start_h2d may re-park them if
                // still full). Each completion frees one slot → guaranteed progress.
                if (!slot_pressure_waiters_.empty()) {
                    auto waiters = std::move(slot_pressure_waiters_);
                    slot_pressure_waiters_.clear();
                    for (const auto& wgk : waiters) {
                        auto wit = gpu_entries_.find(wgk);
                        if (wit == gpu_entries_.end() ||
                            wit->second.tier != GpuTier::kReserved) continue;
                        auto& whe = host_entries_[wgk.key];
                        try_start_h2d(wgk, wit->second, whe);
                    }
                }
            } else {
                ge.transfer_token = 0;
                ge.host_buf_ref.reset();  // TD-82a: release ref (LoadedModel copy kept for retry)
                complete_interests(gk, ge, false, result.lifecycle);
                // complete_interests calls revert_gpu_entry on failure,
                // which publishes kAbsent.
            }
        } else if (xc.direction == transfer::TransferDirection::kD2H) {
            // ELM-6: D2H completion for kDraining entries.
            if (git == gpu_entries_.end() ||
                git->second.tier != GpuTier::kDraining) {
                result.unhandled_transfers.push_back(xc);
                continue;
            }
            auto& ge = git->second;
            auto end_ns = now_ns();

            if (xc.success) {
                // WP-5: write drained data to NVMe file at correct slot offset.
                // Owning overload takes ownership of the buffer unconditionally.
                deps_.nvme_tier->write_expert(gk.key, ge.drain_host_buf);
                ge.drain_host_buf = {};  // ownership transferred
                deps_.expert_cache->evict(gk.key, gk.gpu_idx);

                uint32_t dma_us = elapsed_us(ge.drain_start_ns, end_ns);
                result.lifecycle.push_back({
                    0, ge.drain_cmd_seq, gk.key, gk.gpu_idx,
                    /*success=*/true, /*is_eviction=*/true,
                    /*nvme_us=*/0, dma_us, /*total_us=*/dma_us});
            } else {
                // D2H failed — still evict VRAM (data lost to host).
                deps_.expert_cache->evict(gk.key, gk.gpu_idx);
                deps_.numa_manager->free(ge.drain_host_buf);

                result.lifecycle.push_back({
                    0, ge.drain_cmd_seq, gk.key, gk.gpu_idx,
                    /*success=*/false, /*is_eviction=*/true,
                    0, 0, 0});
            }

            ge.tier = GpuTier::kAbsent;
            ge.transfer_token = 0;
            ge.host_buf_ref.reset();  // Defensive: release packed buffer ref (TD-82j).
            ge.drain_cmd_seq = 0;
            ge.drain_host_buf = {};
            ge.drain_start_ns = 0;
            publish_gpu_state(gk, ge.tier, 0);
        } else {
            result.unhandled_transfers.push_back(xc);
        }
    }

    return result;
}

// ── Zone management ─────────────────────────────────────────────────────────

bool ExpertLifecycleManager::promote(memory::ExpertKey key, int gpu) {
    GpuEntryKey gk{key, gpu};
    auto it = gpu_entries_.find(gk);
    if (it != gpu_entries_.end() && it->second.tier != GpuTier::kHot)
        return false;
    bool ok = deps_.expert_cache->promote(key, gpu);
    if (ok) publish_gpu_state(gk, GpuTier::kHot, 0);
    return ok;
}

bool ExpertLifecycleManager::demote(memory::ExpertKey key, int gpu) {
    GpuEntryKey gk{key, gpu};
    auto it = gpu_entries_.find(gk);
    if (it != gpu_entries_.end() && it->second.tier != GpuTier::kHot)
        return false;
    bool ok = deps_.expert_cache->demote(key, gpu);
    if (ok) publish_gpu_state(gk, GpuTier::kHot, 0);
    return ok;
}

}  // namespace layerstorm::daemon
