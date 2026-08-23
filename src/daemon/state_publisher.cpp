// IPC-5: State publisher implementation.
//
// Copies module state into the flat StateSnapshot struct.  Called once per
// daemon cycle.  Each sub-publish opens its own StateTransaction for the
// seqlock — see spec/IPC_TX.md.

#include "daemon/state_publisher.h"
#include "daemon/state_transaction.h"

#include <algorithm>
#include <cstring>

#include "core/memory/expert_cache.h"
#include "core/memory/page_allocator.h"
#include "core/memory/vram_allocator.h"
#include "core/statistics/acceptance_tracker.h"
#include "core/statistics/expert_stats.h"
#include "core/statistics/workload_detector.h"
#include "core/transfer/transfer_engine.h"

namespace layerstorm::daemon {

StatePublisher::StatePublisher(Deps deps)
    : deps_(std::move(deps)) {
    if (deps_.expert_stats) {
        auto& opts = deps_.expert_stats->options();
        num_moe_layers_  = opts.num_moe_layers;
        num_experts_     = opts.num_experts;
        first_moe_layer_ = opts.first_moe_layer;
    }
    if (deps_.expert_cache) {
        num_gpus_ = deps_.expert_cache->gpu_count();
    } else if (deps_.vram_allocator) {
        num_gpus_ = deps_.vram_allocator->gpu_count();
    }
}

void StatePublisher::publish(ipc::StateSnapshot& snap, StateTransaction& tx) {
    publish_gpu_snapshots(snap, tx);
    // Note: residency_bitmap is now ELM-owned (published event-driven by
    // ExpertLifecycleManager::publish_gpu_state). publish_residency removed.
    publish_expert_stats(snap, tx);
    publish_acceptance(snap, tx);
    publish_workload(snap, tx);
    publish_transfers(snap, tx);
    // Note: expert_host_numa is now ELM-owned (published event-driven by
    // ExpertLifecycleManager::publish_host_state). publish_host_numa removed.
}

// ── Per-GPU state ───────────────────────────────────────────────────────────

void StatePublisher::publish_gpu_snapshots(ipc::StateSnapshot& snap, StateTransaction& tx) {
    StateTransactionGuard guard(tx);
    auto n = static_cast<uint32_t>(std::min(num_gpus_, static_cast<int>(ipc::kMaxGpus)));
    snap.num_gpus = n;

    for (uint32_t g = 0; g < n; ++g) {
        auto& gs = snap.gpus[g];

        if (deps_.vram_allocator) {
            auto& gl = deps_.vram_allocator->layout().gpus[g];
            gs.vram_total_bytes = static_cast<uint64_t>(gl.total_vram_bytes);
            auto& region = deps_.vram_allocator->region(static_cast<int>(g));
            gs.vram_used_bytes = static_cast<uint64_t>(region.allocated_bytes);
        } else {
            gs.vram_total_bytes = 0;
            gs.vram_used_bytes  = 0;
        }

        if (deps_.expert_cache) {
            int gi = static_cast<int>(g);
            gs.expert_stable_used      = static_cast<uint32_t>(
                deps_.expert_cache->used_slots(gi, memory::CacheZone::kStable));
            gs.expert_stable_total     = static_cast<uint32_t>(
                deps_.expert_cache->total_slots(gi, memory::CacheZone::kStable));
            gs.expert_streaming_used   = static_cast<uint32_t>(
                deps_.expert_cache->used_slots(gi, memory::CacheZone::kStreaming));
            gs.expert_streaming_total  = static_cast<uint32_t>(
                deps_.expert_cache->total_slots(gi, memory::CacheZone::kStreaming));
            gs.prefill_mode = static_cast<uint8_t>(
                deps_.expert_cache->prefill_mode(gi));
        } else {
            gs.expert_stable_used    = 0;
            gs.expert_stable_total   = 0;
            gs.expert_streaming_used = 0;
            gs.expert_streaming_total = 0;
            gs.prefill_mode = 0;
        }

        if (deps_.page_allocator) {
            int gi = static_cast<int>(g);
            gs.kv_main_free_pages = static_cast<uint32_t>(
                deps_.page_allocator->free_pages(gi, memory::Pool::kMain));
            gs.kv_spec_free_pages = static_cast<uint32_t>(
                deps_.page_allocator->free_pages(gi, memory::Pool::kSpeculation));
        } else {
            gs.kv_main_free_pages = 0;
            gs.kv_spec_free_pages = 0;
        }

        if (deps_.transfer_engine) {
            int gi = static_cast<int>(g);
            gs.inflight_h2d_count = static_cast<uint32_t>(
                deps_.transfer_engine->inflight_h2d_count(gi));
            gs.inflight_d2h_count = static_cast<uint32_t>(
                deps_.transfer_engine->inflight_d2h_count(gi));
        } else {
            gs.inflight_h2d_count = 0;
            gs.inflight_d2h_count = 0;
        }

        gs.compute_queue_depth = deps_.pending_compute_per_gpu
            ? deps_.pending_compute_per_gpu(g) : 0u;
    }

    // Zero remaining GPU entries.
    for (uint32_t g = n; g < ipc::kMaxGpus; ++g) {
        std::memset(&snap.gpus[g], 0, sizeof(ipc::GpuSnapshot));
    }
}

// Note: publish_residency removed — residency_bitmap is now ELM-owned,
// published event-driven by ExpertLifecycleManager::publish_gpu_state().

// ── Expert statistics ───────────────────────────────────────────────────────

void StatePublisher::publish_expert_stats(ipc::StateSnapshot& snap, StateTransaction& tx) {
    StateTransactionGuard guard(tx);
    size_t active = static_cast<size_t>(num_moe_layers_) * ipc::kMaxExperts;
    size_t active_bytes = active * sizeof(float);

    if (!deps_.expert_stats || active == 0) {
        if (active_bytes > 0) {
            std::memset(snap.expert_frequency, 0, active_bytes);
            std::memset(snap.expert_recency, 0, active_bytes);
            std::memset(snap.expert_routing_weight, 0, active_bytes);
            std::memset(snap.expert_temporal_autocorr, 0, active_bytes);
        }
        return;
    }

    auto& es = *deps_.expert_stats;
    for (uint32_t l = 0; l < num_moe_layers_; ++l) {
        uint32_t abs_layer = l + first_moe_layer_;
        for (uint32_t e = 0; e < num_experts_; ++e) {
            memory::ExpertKey key{abs_layer, static_cast<uint16_t>(e)};
            size_t idx = static_cast<size_t>(l) * ipc::kMaxExperts + e;

            snap.expert_frequency[idx]         = static_cast<float>(es.frequency(key));
            snap.expert_recency[idx]           = static_cast<float>(es.recency(key));
            snap.expert_routing_weight[idx]    = static_cast<float>(es.routing_weight(key));
            snap.expert_temporal_autocorr[idx] = static_cast<float>(es.temporal_autocorr(key));
        }
        // Zero remaining expert slots in this layer (num_experts_ < kMaxExperts).
        for (uint32_t e = num_experts_; e < ipc::kMaxExperts; ++e) {
            size_t idx = static_cast<size_t>(l) * ipc::kMaxExperts + e;
            snap.expert_frequency[idx]         = 0.0f;
            snap.expert_recency[idx]           = 0.0f;
            snap.expert_routing_weight[idx]    = 0.0f;
            snap.expert_temporal_autocorr[idx] = 0.0f;
        }
    }
}

// ── Acceptance tracking ─────────────────────────────────────────────────────

void StatePublisher::publish_acceptance(ipc::StateSnapshot& snap, StateTransaction& tx) {
    StateTransactionGuard guard(tx);
    if (!deps_.acceptance_tracker) {
        snap.global_acceptance_rate     = 0.0;
        snap.windowed_acceptance_rate   = 0.0;
        snap.layer_skip_acceptance_rate = 0.0;
        snap.total_verifications        = 0;
        snap.total_accepted_tokens      = 0;
        snap.total_attempted_tokens     = 0;
        snap.num_tracked_requests       = 0;
        return;
    }

    auto& at = *deps_.acceptance_tracker;
    snap.global_acceptance_rate     = at.global_rate();
    snap.windowed_acceptance_rate   = at.windowed_rate();
    snap.layer_skip_acceptance_rate = at.layer_skip_rate();
    snap.total_verifications        = at.total_verifications();
    snap.total_accepted_tokens      = at.total_accepted_tokens();
    snap.total_attempted_tokens     = at.total_attempted_tokens();

    // Per-request acceptance (IPC-7): snapshot up to kMaxTrackedRequests entries.
    statistics::AcceptanceTracker::PerRequestEntry entries[ipc::kMaxTrackedRequests];
    uint32_t n = at.per_request_snapshot(entries, ipc::kMaxTrackedRequests);
    for (uint32_t i = 0; i < n; ++i) {
        snap.per_request_acceptance[i].request_id      = entries[i].request_id;
        snap.per_request_acceptance[i].acceptance_rate  = entries[i].acceptance_rate;
    }
    snap.num_tracked_requests = n;
}

// ── Workload detector ───────────────────────────────────────────────────────

void StatePublisher::publish_workload(ipc::StateSnapshot& snap, StateTransaction& tx) {
    StateTransactionGuard guard(tx);
    if (!deps_.workload_detector) {
        snap.shift_detected = 0;
        return;
    }
    // One-shot: shift_detected() auto-resets after returning true.
    snap.shift_detected = deps_.workload_detector->shift_detected() ? 1 : 0;
}

// ── Transfer engine ─────────────────────────────────────────────────────────

void StatePublisher::publish_transfers(ipc::StateSnapshot& snap, StateTransaction& tx) {
    StateTransactionGuard guard(tx);
    if (!deps_.transfer_engine) {
        snap.total_inflight_transfers = 0;
        return;
    }
    snap.total_inflight_transfers = static_cast<uint32_t>(
        deps_.transfer_engine->inflight_count());
}

// Note: publish_host_numa removed — expert_host_numa is now ELM-owned,
// published event-driven by ExpertLifecycleManager::publish_host_state().

}  // namespace layerstorm::daemon
