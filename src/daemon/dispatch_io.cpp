// Transfer, NVMe, and fused batch I/O handlers.
// Part of CommandDispatcher — see command_dispatcher.h.

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"
#include "daemon/expert_lifecycle_manager.h"
#include "core/perf_trace.h"

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/memory/expert_cache.h"
#include "core/memory/numa_manager.h"
#include "core/memory/nvme_tier.h"
#include "core/transfer/transfer_engine.h"
#include "daemon/buffer_registry.h"
#include "daemon/spsc_ring.h"

namespace layerstorm::daemon {

static int64_t sub_component_offset(const memory::CacheEntry& entry, uint8_t sub) {
    if (sub == static_cast<uint8_t>(memory::SubComponent::kUp))
        return entry.up_offset;
    if (sub == static_cast<uint8_t>(memory::SubComponent::kDown))
        return entry.down_offset;
    return 0;
}

// ── Transfer handlers ──────────────────────────────────────────────────────

void CommandDispatcher::handle_transfer_h2d(const ipc::Command& cmd) {
    const auto& p = cmd.transfer;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    // Destination: VRAM slot from ExpertCache (must be previously reserved).
    const auto* entry = deps_.expert_cache->lookup(key, gpu);
    if (!entry || !entry->vram_address) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kCacheSlotMissing,
                    "H2D: no cache reservation for expert");
        return;
    }

    // Apply sub-component offset for partial transfers.
    int64_t offset = sub_component_offset(*entry, p.sub_component);
    void* dst = static_cast<uint8_t*>(entry->vram_address) + offset;

    // Source: host-side expert weight data + same offset.
    auto hs = resolve_host_source(p.layer_idx, p.expert_idx);
    const void* src_base = hs.ptr;
    auto host_buf = hs.owned_buf;
    bool pinned = hs.pinned;
    if (!src_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kHostSourceMissing,
                    "H2D: no host source for expert");
        return;
    }
    const void* src = static_cast<const uint8_t*>(src_base) + offset;

    // Enqueue the transfer.  Callback marks sub-components ready.
    // TD-82a: capture host_buf shared_ptr to keep buffer alive during DMA.
    auto sub = p.sub_component;
    auto token = deps_.transfer_engine->enqueue_h2d(
        key, gpu, dst, src, p.bytes,
        p.priority, static_cast<int>(p.delay_us),
        [this, key, gpu, sub,
         host_buf = std::move(host_buf)](transfer::TransferToken /*tok*/, bool success) {
            if (success && deps_.expert_cache) {
                deps_.expert_cache->mark_ready(
                    key, gpu,
                    static_cast<memory::SubComponent>(sub));
            }
            // host_buf drops here, releasing TD-82a reference.
        },
        /*needs_pinned_staging=*/!pinned);

    if (token.has_value()) {
        register_pcie_token(token.value(), cmd.cmd_seq);
    }
    // nullopt = dedup (already in-flight) — no error, no completion yet.
}

void CommandDispatcher::handle_transfer_d2h(const ipc::Command& cmd) {
    const auto& p = cmd.transfer;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(cmd.gpu_idx);

    // Source: VRAM slot.
    const auto* entry = deps_.expert_cache->lookup(key, gpu);
    if (!entry || !entry->vram_address) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kCacheSlotMissing,
                    "D2H: expert not resident on GPU");
        return;
    }

    // Apply sub-component offset for partial transfers.
    int64_t offset = sub_component_offset(*entry, p.sub_component);
    const void* src = static_cast<const uint8_t*>(entry->vram_address) + offset;

    // Destination: host-side buffer from NvmeTier.
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeTierMissing,
                    "D2H: NVMe tier not available");
        return;
    }
    // host_ptr() returns the warm cache buffer for this expert.
    const void* host = deps_.nvme_tier->host_ptr(key);
    if (!host) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeTierMissing,
                    "D2H: no host buffer for expert");
        return;
    }
    // D2H destination is mutable, with same sub-component offset.
    void* dst = const_cast<uint8_t*>(static_cast<const uint8_t*>(host)) + offset;

    auto token = deps_.transfer_engine->enqueue_d2h(
        key, gpu, dst, src, p.bytes,
        p.priority, static_cast<int>(p.delay_us));

    if (token.has_value()) {
        register_pcie_token(token.value(), cmd.cmd_seq);
    }
}

// ── Host source resolution ─────────────────────────────────────────────────

CommandDispatcher::HostSourceResult CommandDispatcher::resolve_host_source(
        uint32_t layer_idx, uint16_t expert_idx) {
    return transfer::resolve_expert_host_source(
        make_key(layer_idx, expert_idx), host_source_deps_);
}

// ── NVMe tier handlers (IPC-8a.1) ────────────────────────────────────────

void CommandDispatcher::handle_nvme_read(const ipc::Command& cmd) {
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeRead,
                    "nvme_read: NVMe tier not configured");
        return;
    }

    const auto& p = cmd.nvme_read;
    auto key = make_key(p.layer_idx, p.expert_idx);

    auto token = deps_.nvme_tier->read_expert(
        key, static_cast<int>(p.gpu_hint));

    if (token.has_value()) {
        register_nvme_token(token.value(), cmd.cmd_seq);
    } else {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeRead,
                    "nvme_read: read_expert failed "
                    "(not on NVMe, already warm, inflight, or ring full)");
    }
}

void CommandDispatcher::handle_nvme_write(const ipc::Command& cmd) {
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeWrite,
                    "nvme_write: NVMe tier not configured");
        return;
    }

    const auto& p = cmd.nvme_write;
    auto key = make_key(p.layer_idx, p.expert_idx);

    // Expert must be in host RAM warm cache to write to NVMe.
    const void* host_ptr = deps_.nvme_tier->host_ptr(key);
    if (!host_ptr) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeWrite,
                    "nvme_write: expert not in host RAM warm cache");
        return;
    }

    auto token = deps_.nvme_tier->write_expert(key, host_ptr);

    if (token.has_value()) {
        register_nvme_token(token.value(), cmd.cmd_seq);
    } else {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeWrite,
                    "nvme_write: write_expert failed "
                    "(inflight, ring full, or io_uring unavailable)");
    }
}

void CommandDispatcher::handle_nvme_evict_host(const ipc::Command& cmd) {
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeEvict,
                    "nvme_evict_host: NVMe tier not configured");
        return;
    }

    const auto& p = cmd.nvme_evict_host;
    auto key = make_key(p.layer_idx, p.expert_idx);

    // ELM guard: refuse discard while NVMe read inflight for this expert.
    if (deps_.elm &&
        deps_.elm->host_state(key) == HostTier::kLoadingToRam) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeEvict,
                    "nvme_evict_host: expert loading to RAM (ELM)");
        return;
    }

    // WP-5: mmap-backed NvmeTier — mmap has process lifetime, no explicit
    // discard needed.  The OS page cache manages residency.  Report success.
    (void)key;
    write_nvme_completion(cmd.cmd_seq, cmd.gpu_idx,
                          p.layer_idx, p.expert_idx,
                          /*op=*/2, /*status=*/0);
}

void CommandDispatcher::handle_cancel_transfer(const ipc::Command& cmd) {
    const uint32_t target = cmd.cancel_transfer.target_cmd_seq;

    // ELM path: if target belongs to an ELM-managed expert, cancel through ELM.
    // Uses find_all_by_cmd_seq to handle batch commands (shared cmd_seq).
    if (deps_.elm) {
        auto tokens = deps_.elm->find_all_by_cmd_seq(target);
        if (!tokens.empty()) {
            for (auto lt : tokens) {
                deps_.elm->cancel(lt);
            }
            write_cancel_completion(cmd.cmd_seq, cmd.gpu_idx, target,
                                    /*cancelled=*/1, /*status=*/0);
            return;
        }
    }

    // Legacy path: non-ELM transfers.
    auto it = cmd_seq_to_token_.find(target);
    if (it == cmd_seq_to_token_.end()) {
        write_cancel_completion(cmd.cmd_seq, cmd.gpu_idx, target,
                                /*cancelled=*/0, /*status=*/0);
        return;
    }

    auto [internal_token, is_nvme] = it->second;

    if (is_nvme) {
        // Cancel NVMe I/O.
        deps_.nvme_tier->cancel(internal_token);
        nvme_token_to_cmd_seq_.erase(internal_token);
    } else {
        // Cancel PCIe transfer.
        auto cancelled_info = deps_.transfer_engine->cancel(internal_token);
        token_to_cmd_seq_.erase(internal_token);

        // For H2D: evict the incomplete VRAM slot (reserved but never
        // mark_ready).  The data is stale.
        if (cancelled_info &&
            cancelled_info->direction == transfer::TransferDirection::kH2D &&
            deps_.expert_cache) {
            // TD-EVICT-BOARD-DESYNC: ExpertCache::evict fires the residency
            // listener (the EvictScoreBoard) for stable-zone slots, covering this
            // cancel / never-arrived case automatically — no manual board feed.
            deps_.expert_cache->evict(cancelled_info->key,
                                      cancelled_info->gpu_idx);
        }
    }

    cmd_seq_to_token_.erase(it);
    write_cancel_completion(cmd.cmd_seq, cmd.gpu_idx, target,
                            /*cancelled=*/1, /*status=*/0);
}

// ── Fused command handlers (IPC-8e) ──────────────────────────────────────

void CommandDispatcher::handle_prefetch_expert(const ipc::Command& cmd) {
    const auto& p = cmd.prefetch_expert;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(p.gpu_idx);

    // ELM path: auto-chains from any tier (NVMe→RAM→VRAM).
    if (deps_.elm) {
        deps_.elm->ensure_resident(key, gpu, to_zone(p.zone), cmd.cmd_seq,
                                   p.priority, static_cast<int>(p.delay_us));
        // Completion delivered by DaemonLoop via elm->poll().
        return;
    }

    // Legacy path (no ELM): direct reserve + H2D.
    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kPrefetchExpert,
                    "prefetch_expert: expert cache not configured");
        return;
    }
    if (!deps_.transfer_engine) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kPrefetchExpert,
                    "prefetch_expert: transfer engine not configured");
        return;
    }

    void* addr = deps_.expert_cache->reserve(
        key, gpu, to_zone(p.zone), /*is_duplicate=*/false);
    if (!addr) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kCacheReserveFailed,
                    "prefetch_expert: cache reserve failed");
        return;
    }

    auto hs = resolve_host_source(p.layer_idx, p.expert_idx);
    const void* src = hs.ptr;
    auto host_buf = hs.owned_buf;
    bool pinned = hs.pinned;
    if (!src) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kHostSourceMissing,
                    "prefetch_expert: no host source for expert");
        return;
    }

    // GG-9 (TD-GG9-EXPERT-PERLAYER-COPYSIZE): copy the actual host source size
    // (a mixed "XL" GGUF packs each layer's experts smaller than the global-max
    // slot); copying the full slot would overrun the smaller host buffer.
    auto bytes = deps_.expert_cache->expert_bytes();
    if (hs.bytes > 0 && hs.bytes < bytes) bytes = hs.bytes;
    auto sub = static_cast<uint8_t>(memory::SubComponent::kAll);
    auto token = deps_.transfer_engine->enqueue_h2d(
        key, gpu, addr, src, static_cast<int64_t>(bytes),
        p.priority, static_cast<int>(p.delay_us),
        [this, key, gpu, sub,
         host_buf = std::move(host_buf)](transfer::TransferToken, bool success) {
            if (success && deps_.expert_cache) {
                deps_.expert_cache->mark_ready(
                    key, gpu,
                    static_cast<memory::SubComponent>(sub));
            }
        },
        /*needs_pinned_staging=*/!pinned);

    if (token.has_value()) {
        register_pcie_token(token.value(), cmd.cmd_seq);
    }
}

void CommandDispatcher::handle_evict_to_host(const ipc::Command& cmd) {
    const auto& p = cmd.evict_to_host;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(p.gpu_idx);

    if (deps_.elm) {
        bool ok = deps_.elm->request_evict(key, gpu);
        write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
        return;
    }

    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kReserveBatch,
                    "evict_to_host: expert cache not configured");
        return;
    }

    // TD-EVICT-BOARD-DESYNC: ExpertCache::evict fires the residency listener.
    bool ok = deps_.expert_cache->evict(key, gpu);
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
}

void CommandDispatcher::handle_slow_evict_to_host(const ipc::Command& cmd) {
    const auto& p = cmd.slow_evict_to_host;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(p.gpu_idx);

    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kElmExpertOp,
                    "slow_evict: expert cache not configured");
        return;
    }

    // Check VRAM residency.
    const auto* entry = deps_.expert_cache->lookup(key, gpu);
    if (!entry) {
        write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 1u);  // not resident
        return;
    }

    // Check if host already has data — fast path (metadata-only evict).
    bool host_has_data = false;
    if (deps_.elm) {
        host_has_data = (deps_.elm->host_state(key) == HostTier::kWarm);
    } else {
        // Existence check only — don't trigger lazy packing (TD-82a).
        host_has_data = (resolve_host_source(p.layer_idx, p.expert_idx).ptr != nullptr);
    }

    if (host_has_data) {
        if (deps_.elm) {
            bool ok = deps_.elm->request_evict(key, gpu);
            write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
        } else {
            // TD-EVICT-BOARD-DESYNC: ExpertCache::evict fires the listener.
            bool ok = deps_.expert_cache->evict(key, gpu);
            write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, ok ? 0u : 1u);
        }
        return;
    }

    // Slow path: D2H to host, then evict from VRAM.

    // ELM-6: delegate to ELM's request_drain when available.
    // Completion arrives via ELM poll → CMP_ELM_EXPERT_EVICTED.
    if (deps_.elm) {
        bool ok = deps_.elm->request_drain(key, gpu, cmd.cmd_seq);
        if (!ok) {
            write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 1u);
        }
        return;
    }

    // Legacy fallback (no ELM): callback-based D2H + evict.
    if (!deps_.transfer_engine) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kElmExpertOp,
                    "slow_evict: transfer engine not available");
        return;
    }
    if (!deps_.nvme_tier || !deps_.numa_manager) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kElmExpertOp,
                    "slow_evict: NVMe tier or NUMA manager not available");
        return;
    }

    auto bytes = deps_.expert_cache->expert_bytes();
    auto host_buf = deps_.numa_manager->allocate_for_gpu(
        static_cast<size_t>(bytes), gpu);

    auto cmd_seq = cmd.cmd_seq;
    auto gpu_idx = cmd.gpu_idx;

    auto* cache    = deps_.expert_cache;
    auto* nvme     = deps_.nvme_tier;
    auto* numa     = deps_.numa_manager;
    auto* cmp_ring = deps_.cmp_ring;

    auto token = deps_.transfer_engine->enqueue_d2h(
        key, gpu, host_buf.data, entry->vram_address, bytes,
        0.0f, 0,
        [=, this](transfer::TransferToken /*tok*/, bool success) mutable {
            if (success) {
                // WP-5: write drained data to NVMe file (owning overload).
                nvme->write_expert(key, host_buf);
                host_buf = {};  // ownership transferred
                cache->evict(key, gpu);  // fires the residency listener (TD-EVICT-BOARD-DESYNC)
                ipc::Completion cmp{};
                cmp.cmp_type = ipc::CMP_CACHE_OP_DONE;
                cmp.cmd_seq  = cmd_seq;
                cmp.gpu_idx  = gpu_idx;
                cmp.status   = 0u;
                if (!cmp_ring->try_write(&cmp)) {
                    spdlog::error("CommandDispatcher: completion ring full "
                                  "(slow_evict done)");
                }
            } else {
                numa->free(host_buf);
                ipc::Completion cmp{};
                cmp.cmp_type = ipc::CMP_CACHE_OP_DONE;
                cmp.cmd_seq  = cmd_seq;
                cmp.gpu_idx  = gpu_idx;
                cmp.status   = 1u;
                if (!cmp_ring->try_write(&cmp)) {
                    spdlog::error("CommandDispatcher: completion ring full "
                                  "(slow_evict fail)");
                }
            }
        });

    if (!token.has_value()) {
        numa->free(host_buf);
        write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 1u);
    }
}

void CommandDispatcher::handle_stage_expert(const ipc::Command& cmd) {
    const auto& p = cmd.stage_expert;
    auto key = make_key(p.layer_idx, p.expert_idx);
    auto gpu = static_cast<int>(p.gpu_idx);

    // ELM path: ensure_resident auto-detects cold tier and chains NVMe→RAM→VRAM.
    if (deps_.elm) {
        deps_.elm->ensure_resident(key, gpu, to_zone(p.zone), cmd.cmd_seq,
                                   p.priority, static_cast<int>(p.delay_us));
        return;
    }

    // Legacy path: NVMe read only, no auto-chain.
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeBatchRead,
                    "stage_expert: NVMe tier not configured");
        return;
    }

    auto token = deps_.nvme_tier->read_expert(key, gpu);
    if (token.has_value()) {
        register_nvme_token(token.value(), cmd.cmd_seq);
    } else {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeBatchRead,
                    "stage_expert: NVMe read failed");
    }
}

void CommandDispatcher::handle_prefetch_batch(const ipc::Command& cmd) {
    perf_trace::record(perf_trace::kCmdDrained,
                       static_cast<uint16_t>(cmd.gpu_idx), cmd.cmd_seq, 0, 0);
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kPrefetchBatch,
                    "prefetch_batch: sideband not configured");
        return;
    }

    const auto& p = cmd.prefetch_batch;
    if (p.count == 0 || p.count > ipc::kMaxExpertPrefetch) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kPrefetchBatch,
                    "prefetch_batch: count out of range");
        return;
    }

    const auto* entries = reinterpret_cast<const ipc::ExpertPrefetchEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertPrefetchOff);

    // ELM path: delegate each entry to ensure_resident.
    if (deps_.elm) {
        for (uint32_t i = 0; i < p.count; ++i) {
            const auto& e = entries[i];
            auto key = make_key(e.layer_idx, e.expert_idx);
            deps_.elm->ensure_resident(key, static_cast<int>(e.gpu_idx),
                                       to_zone(e.zone), cmd.cmd_seq,
                                       p.priority, static_cast<int>(p.delay_us));
        }
        return;
    }

    // Legacy path: direct reserve + H2D per entry.
    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kPrefetchBatch,
                    "prefetch_batch: expert cache not configured");
        return;
    }

    uint32_t enqueued = 0;
    for (uint32_t i = 0; i < p.count; ++i) {
        const auto& e = entries[i];
        auto key = make_key(e.layer_idx, e.expert_idx);
        auto gpu = static_cast<int>(e.gpu_idx);

        void* addr = deps_.expert_cache->reserve(
            key, gpu, to_zone(e.zone), /*is_duplicate=*/false);
        if (!addr) continue;

        auto hs = resolve_host_source(e.layer_idx, e.expert_idx);
        const void* src = hs.ptr;
        auto host_buf = hs.owned_buf;
        bool pinned = hs.pinned;
        if (!src) continue;

        // GG-9 (TD-GG9-EXPERT-PERLAYER-COPYSIZE): copy the actual source size,
        // not the global-max slot (would overrun a smaller per-layer host buffer).
        auto bytes = deps_.expert_cache->expert_bytes();
        if (hs.bytes > 0 && hs.bytes < bytes) bytes = hs.bytes;
        auto sub = static_cast<uint8_t>(memory::SubComponent::kAll);
        auto token = deps_.transfer_engine->enqueue_h2d(
            key, gpu, addr, src, static_cast<int64_t>(bytes),
            p.priority, static_cast<int>(p.delay_us),
            [this, key, gpu, sub,
             host_buf = std::move(host_buf)](transfer::TransferToken, bool success) {
                if (success && deps_.expert_cache) {
                    deps_.expert_cache->mark_ready(
                        key, gpu,
                        static_cast<memory::SubComponent>(sub));
                }
            },
            /*needs_pinned_staging=*/!pinned);

        if (token.has_value()) {
            if (enqueued == 0) {
                register_pcie_token(token.value(), cmd.cmd_seq);
            } else {
                token_to_cmd_seq_[token.value()] = cmd.cmd_seq;
            }
            ++enqueued;
        }
    }

    if (enqueued == 0) {
        write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
    }
}

void CommandDispatcher::handle_evict_batch(const ipc::Command& cmd) {
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kEvictBatch,
                    "evict_batch: sideband not configured");
        return;
    }

    const auto& p = cmd.evict_batch;
    if (p.count == 0 || p.count > ipc::kMaxExpertEviction) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kEvictBatch,
                    "evict_batch: count out of range");
        return;
    }

    const auto* entries = reinterpret_cast<const ipc::ExpertEvictionEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertEvictionOff);

    if (deps_.elm) {
        for (uint32_t i = 0; i < p.count; ++i) {
            const auto& e = entries[i];
            deps_.elm->request_evict(make_key(e.layer_idx, e.expert_idx),
                                     static_cast<int>(e.gpu_idx));
        }
        write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
        return;
    }

    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kEvictBatch,
                    "evict_batch: expert cache not configured");
        return;
    }

    for (uint32_t i = 0; i < p.count; ++i) {
        const auto& e = entries[i];
        const auto k = make_key(e.layer_idx, e.expert_idx);
        const int g = static_cast<int>(e.gpu_idx);
        // TD-EVICT-BOARD-DESYNC: ExpertCache::evict fires the residency listener.
        deps_.expert_cache->evict(k, g);
    }
    write_cache_completion(cmd.cmd_seq, cmd.gpu_idx, 0);
}

void CommandDispatcher::handle_nvme_batch_read(const ipc::Command& cmd) {
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeBatchReadNew,
                    "nvme_batch_read: sideband not configured");
        return;
    }
    if (!deps_.nvme_tier) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeBatchReadNew,
                    "nvme_batch_read: NVMe tier not configured");
        return;
    }

    const auto& p = cmd.nvme_batch_read;
    if (p.count == 0 || p.count > ipc::kMaxNvmeReadBatch) {
        write_error(cmd.cmd_seq, cmd.gpu_idx, ipc::CmpErrorCategory::kNvmeBatchReadNew,
                    "nvme_batch_read: count out of range");
        return;
    }

    const auto* entries = reinterpret_cast<const ipc::NvmeReadEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kNvmeReadOff);

    uint32_t submitted = 0;
    for (uint32_t i = 0; i < p.count; ++i) {
        const auto& e = entries[i];
        auto key = make_key(e.layer_idx, e.expert_idx);

        auto token = deps_.nvme_tier->read_expert(
            key, static_cast<int>(e.gpu_hint));

        if (token.has_value()) {
            nvme_token_to_cmd_seq_[token.value()] = cmd.cmd_seq;
            if (submitted == 0) {
                register_nvme_token(token.value(), cmd.cmd_seq);
            }
            ++submitted;
        }
    }

    if (submitted == 0) {
        write_nvme_completion(cmd.cmd_seq, cmd.gpu_idx, 0, 0, 0, 0);
    }
}

}  // namespace layerstorm::daemon
