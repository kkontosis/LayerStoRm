// Progressive fetch-and-run MoE (#90) — FETCH_AND_RUN_MOE handling, wave
// scheduling, FAR stream gate, overlap passes, and transient-zone sweep.
// Part of CommandDispatcher — see command_dispatcher.h. Split from
// dispatch_moe.cpp (pure code motion).

#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"
#include "daemon/ep_residency_dedup.h"  // INV-MOE-EP-DISJOINT force-ON dedup
#include "daemon/expert_lifecycle_manager.h"
#include "core/perf_trace.h"

#include <algorithm>
#include <atomic>   // TD-DRIFT-ROOTCAUSE diagnostic dump sequence counter
#include <chrono>
#include <cstdio>  // I8b shadow-dump JSONL sink (fopen/fprintf)
#include <cstdlib>  // TD-DRIFT-ROOTCAUSE getenv gate
#include <cstring>
#include <limits>
#include <span>
#include <string>  // I8 shadow log assembly
#include <unordered_map>  // TD-PREFILL-FETCH-SEAM-SCALING wave scheduling
#include <unordered_set>  // TD-FAR-SLOT-RESERVE-STALL needed-key set
#include <vector>  // TD-DRIFT-ROOTCAUSE D2H staging

#include <spdlog/spdlog.h>

#include "compute/stream_manager.h"
#include "core/device_backend.h"
#include "core/expert_device.h"
#include "core/gpu_loader/far_evict.h"  // shared reroute-eviction fallback
#include "core/memory/expert_cache.h"
#include "core/transfer/transfer_engine.h"  // TD-FAR-STREAM-GATE: h2d barrier
#include "core/memory/numa_manager.h"  // I8 shadow: expert_home_node()
#include "model/quantization/gguf_kquant.h"  // GG-5b: GgufQuantInterface (per-projection type)
#include "core/statistics/expert_stats.h"  // ExpertStats recency feed (FETCH path)
#include "parallelism/dcp_communicator.h"
#include "compute/kernels/elementwise/residual_add.h"
#include "compute/kernels/mhc/mhc.h"
#include "smxx/permute/moe_permute.h"  // DET-REDUCE Phase 1b: fp32→bf16 EP-combine cast
#include "compute/kernels/moe/moe_gemm_meta.h"
#include "compute/kernels/moe/router_projection.h"
#include "compute/kernels/norm/rmsnorm.h"
#include "compute/kernels/moe/hash_gating.h"  // V4-4 hash-layer gating
#include "compute/kernels/sm120/gemm/bf16_gemm.h"  // V4-7b raw-BF16 shexp/dense FFN
#include "sm120/gating/topk_gating.h"

namespace layerstorm::daemon {

namespace {
// Demand-fetch priority for the E_CMD_FETCH_AND_RUN_MOE internal H2D path. The
// MoE GEMM is stalled waiting on these experts (they sit fully on the decode
// critical path), so they must outrank any speculative prefetch already staged
// in the transfer engine's per-GPU priority queue. Using max() guarantees the
// demand fetch is admitted first regardless of the scale the orchestrator uses
// for PREFETCH_BATCH priorities. Note: this only reorders the *staged* admission
// queue — it cannot cancel an already-dispatched in-flight DMA (those are
// uncancelable); reserving inflight headroom for demand is tracked as TD-TRANS-hw.
// Applied ONLY on the E_CMD_FETCH_AND_RUN_MOE fetch; orchestrator-issued
// PREFETCH_BATCH keeps its command-supplied priority (default 0).
constexpr float kDemandFetchPriority = std::numeric_limits<float>::max();

// TD-FAR-SLOT-RESERVE-STALL: default-ON kill-switch for the batched inline-evict
// residency path (ensure_residents). LS_FAR_ENSURE_RESIDENTS=0 restores the legacy
// per-expert ensure_resident + one-shot apply_far_evictions make-room (A/B + revert).
static bool far_ensure_residents_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_FAR_ENSURE_RESIDENTS");
        return !(v && v[0] == '0');
    }();
    return on;
}
}  // namespace

// ── #90: Progressive fetch-and-run MoE ─────────────────────────────────────

void CommandDispatcher::ProgressiveMoeState::release_locks(
        memory::ExpertCache* cache) {
    if (!cache) return;
    for (auto& er : experts) {
        if (er.is_locked) {
            cache->unlock(er.key, er.target_gpu);
            er.is_locked = false;
        }
    }
}


void CommandDispatcher::handle_fetch_and_run_moe(const ipc::Command& cmd) {
    handle_fetch_and_run_moe_impl(cmd, /*big=*/false);
}

// Shared body of E_CMD_FETCH_AND_RUN_MOE and E_CMD_FETCH_AND_RUN_MOE_BIG.
// The BIG payload is a layout-compatible prefix extension of the legacy one
// (common initial sequence), so both are read through cmd.fetch_and_run_moe;
// only chunk_tokens is BIG-specific.
void CommandDispatcher::handle_fetch_and_run_moe_impl(const ipc::Command& cmd,
                                                      bool big) {
    const auto& p = cmd.fetch_and_run_moe;
    perf_trace::record(perf_trace::kMoeEnter,
                       static_cast<uint16_t>(cmd.gpu_idx), cmd.cmd_seq,
                       p.layer_idx, 0);

    // Only one progressive MoE at a time — but a queued (burst-published)
    // FETCH is backpressure, not an error: drain the active one first.
    if (!drain_progressive_moe("fetch_and_run_moe")) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "progressive MoE already active");
        return;
    }

    if (!deps_.expert_cache) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "expert cache not configured");
        return;
    }

    // TD-PREFILL-SUPERCHUNK: the MoE pipeline scratch is sized for
    // moe_batch_capacity_ tokens — a larger num_seqs would overflow
    // permuted_input/wave_accum/topk. Fail loud, never overflow.
    if (p.num_seqs == 0
        || p.num_seqs > static_cast<uint32_t>(moe_batch_capacity_)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "fetch_and_run_moe: num_seqs exceeds MoE batch capacity");
        return;
    }

    // ── Build progressive state ────────────────────────────────────────

    ProgressiveMoeState state;
    state.cmd_seq   = cmd.cmd_seq;
    state.gpu_idx   = cmd.gpu_idx;
    state.layer_idx = p.layer_idx;
    state.num_seqs  = p.num_seqs;
    state.moe_mode  = p.moe_mode;
    // E_CMD_FAR_FORWARD_LAYER delegation: the fused handler synthesizes this
    // command with ITS cmd_type (payload read through the union is layout-
    // identical). Set the completion overrides HERE — the num_seqs==1
    // gated-final path (LS_FAR_GATED_FINAL, default ON) finalizes INSIDE
    // this call, before any post-return patching could run.
    if (cmd.cmd_type == ipc::E_CMD_FAR_FORWARD_LAYER) {
        state.cmp_cmd_type_override = cmd.cmd_type;
        state.cmp_data_bytes        = p.expert_count;
    }
    state.total_experts = static_cast<int>(p.expert_count);
    // TD-PREFILL-MOE-BIG: BIG command — double-buffered waves + chunked
    // execution (chunk override clamped inside the chunked dispatch).
    state.big = big;
    state.chunk_tokens = big
        ? static_cast<int>(cmd.fetch_and_run_moe_big.chunk_tokens) : 0;

    // Read sideband expert list.
    const auto* entries = reinterpret_cast<const ipc::ExpertPrefetchEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertPrefetchOff);

    // F-6: per-entry gating weights are carried inline only when the fetch list
    // fits the decode top-K cap; otherwise weight-threshold gating is inactive
    // (count-based deciders still apply).
    const bool have_weights =
        p.weight_count > 0 && p.expert_count <= ipc::kMaxFetchDeciderWeights;

    state.experts.resize(p.expert_count);
    for (uint32_t i = 0; i < p.expert_count; ++i) {
        const auto& e = entries[i];
        auto key = make_key(e.layer_idx, e.expert_idx);
        int target = static_cast<int>(e.gpu_idx);

        auto& er = state.experts[i];
        er.key = key;
        er.target_gpu = target;
        er.zone = e.zone;  // TD-FAR-GATING: keep for ensure_resident re-issue
        er.weight = (have_weights && i < p.weight_count) ? p.weights[i] : 0.0f;

        // C-6 Milestone A: a forced CPU expert bypasses ALL GPU machinery — it
        // is never made resident / fetched / locked / evicted / GPU-computed,
        // and stays out of every GPU's routed bitset (is_arrived=false ⇒ the
        // finalize bitset loop excludes it). The host computes it from the
        // pinned arena and folds the contribution post-EP-combine
        // (fold_cpu_forced_experts). Counted so the phase check does not wait
        // on it, but NOT as a miss.
        // TASK-2: in the never-lose CPU-solver bridge (LS_LOADER_CPU_SOLVER) the
        // SOLVER decides CPU offload per token AFTER this state-build (route_moe_by_
        // loader, below); the static map here would be STALE by one token. Gate the
        // static marking OFF in solver mode — route is the sole authority. The
        // static LS_CPU_EXPERT_FORCE path is unaffected (loader_cpu_solver_ false).
        if (!loader_cpu_solver_ && is_cpu_forced(e.layer_idx, e.expert_idx)) {
            er.cpu_forced = true;
            ++state.cpu_forced_count;
            continue;
        }

        // Check residency.
        const auto* ce = deps_.expert_cache->lookup(key, target);
        if (ce && (ce->sub_components_ready & memory::SubComponent::kAll)
                    == memory::SubComponent::kAll) {
            er.was_cached = true;
            er.is_arrived = true;
            ++state.arrived_count;
        }
    }

    // I8 GPU-loader shadow solve + (LS_LOADER_ACT) reroute. Env-gated, inert in
    // production; implemented in dispatch_loader.cpp.
    route_moe_by_loader(entries, cmd, state);

    // ── F-6: selective-fetch decider ───────────────────────────────────
    // Decide which *missing* (non-resident) experts to issue H2D for. Resident
    // experts are always included (free). Entries are in gating SELECTION-RANK
    // order (the gating top-K order; NOT weight-sorted — V3.2 selects by biased
    // score but `weight` is the unbiased sigmoid score, INV-10c-2). Honoring
    // min_experts/max_new_fetches by position keeps the model's top-ranked picks.
    // A missing expert is fetched iff:
    //   (its index < min_experts)  OR  (weight gating passes), and
    //   the running H2D-issue count is below max_new_fetches (0 = unlimited).
    // Missing experts not selected are marked skipped → graceful degradation
    // identical to the timeout path (excluded from the bitset).
    {
        const uint32_t max_new = p.max_new_fetches;   // 0 = unlimited
        const uint32_t min_exp = p.min_experts;       // 0 = no floor
        const float    thresh  = p.gating_weight_threshold;
        const bool     gate_by_weight = have_weights && thresh > 0.0f;
        uint32_t issued = 0;
        for (uint32_t i = 0; i < p.expert_count; ++i) {
            auto& er = state.experts[i];
            if (er.cpu_forced) continue;  // C-6: host-computed, no GPU fetch
            if (er.is_arrived) continue;  // resident — always included

            const bool forced = (i < min_exp);
            // Weight gating: when active, a sub-threshold non-forced expert is
            // skipped. When inactive (no weights / no positive cutoff), all pass.
            const bool weight_ok = !gate_by_weight || er.weight >= thresh;
            bool select = forced || weight_ok;

            // max_new_fetches cap (0 = unlimited). Forced experts (min_experts)
            // are always honored and do not consume the cap budget.
            if (select && !forced && max_new > 0 && issued >= max_new) {
                select = false;
            }

            if (select) {
                er.fetch_requested = true;
                if (!forced) ++issued;
            } else {
                ++state.skipped_count;
            }
        }
    }

    // ── TD-CHUNK-SMALLM-DEFAULT / union-aware cache partitioning ───────
    // TRANSIENT fetch class: sideband entries with zone=1 (streaming) are
    // scan-resistant union fetches — they reserve in the expert_streaming
    // zone (spill→prefetch sub-allocators), never trigger stable make-room
    // (see the zone==0 guard in apply_far_evictions' is_real_fetch below),
    // and every evictable streaming resident on the involved GPUs is
    // RELEASED when this command's completion is reaped (sweep-on-reap,
    // dispatch_completion.cpp — after the kExpertFfn event, so the GPU is
    // done reading the weights). No production/test path sends zone=1 FETCH
    // entries unless explicitly armed (dsp52 DSP52_UPART=1), so zone=0
    // commands are byte-identical.
    //
    // CAPACITY CLAMP (0cf5823a discipline: no fabricated budgets, reserve
    // feasibility first): per target GPU, transient fetches beyond the free
    // streaming-slot count fall back to zone=0 — today's exact stable
    // demand path (victim-map / board-fallback make-room) — instead of a
    // guaranteed-to-fail streaming reserve (drop → re-drive livelock).
    if (deps_.cuda_kernels_enabled && deps_.expert_cache) {
        std::unordered_map<int, int> stream_budget;
        int flipped = 0;
        for (auto& er : state.experts) {
            if (er.zone == 0) continue;
            if (er.is_arrived || !er.fetch_requested) {
                // Resident (or skipped) transient entry: no slot needed, but
                // the sweep must still cover its GPU (a prefetched streaming
                // copy served it and is released after use).
                state.transient_gpu_mask |= 1u << (er.target_gpu & 31);
                continue;
            }
            auto it = stream_budget.find(er.target_gpu);
            if (it == stream_budget.end())
                it = stream_budget.emplace(
                    er.target_gpu,
                    deps_.expert_cache->free_slots(
                        er.target_gpu, memory::CacheZone::kStreaming)).first;
            if (it->second > 0) {
                --it->second;
                state.transient_gpu_mask |= 1u << (er.target_gpu & 31);
            } else {
                er.zone = 0;  // overflow → today's stable behavior
                ++flipped;
            }
        }
        if (flipped > 0)
            spdlog::debug("fetch_and_run_moe: {} transient (zone=1) fetch(es) "
                          "exceeded free streaming slots — fell back to the "
                          "stable zone (layer {}, cmd_seq {})",
                          flipped, p.layer_idx, cmd.cmd_seq);
    }

    // ── TD-ORCH-ELM-COMPLETION-LIVELOCK: 0-capacity target guard ───────
    // A fetch entry targeting a GPU whose stable zone has ZERO total slots
    // (e.g. a draft/XTP GPU outside the orchestrator's EP owner set, or a
    // misconfigured topology) can NEVER become resident there — issuing it
    // would only produce failed lifecycle completions and burn the command
    // deadline. Fail loud ONCE per command and degrade exactly like the
    // decider's skipped experts (excluded from the bitset, counted in
    // routed_miss_count).
    if (deps_.cuda_kernels_enabled) {
        int zero_cap_skipped = 0;
        int first_bad_gpu = -1;
        memory::ExpertKey first_bad_key{};
        for (auto& er : state.experts) {
            if (er.is_arrived || !er.fetch_requested || er.zone != 0) continue;
            if (deps_.expert_cache->total_slots(
                    er.target_gpu, memory::CacheZone::kStable) > 0) continue;
            er.fetch_requested = false;
            ++state.skipped_count;
            if (zero_cap_skipped == 0) {
                first_bad_gpu = er.target_gpu;
                first_bad_key = er.key;
            }
            ++zero_cap_skipped;
        }
        if (zero_cap_skipped > 0) {
            spdlog::error(
                "fetch_and_run_moe: {} fetch entr{} target GPU(s) with a "
                "0-slot stable zone (first: L{}E{} gpu={}) — skipped "
                "(degraded), NOT retried (cmd_seq {}, layer {}). The "
                "orchestrator's EP owner set (ep_gpu_indices) must exclude "
                "GPUs hosting no expert cache.",
                zero_cap_skipped, zero_cap_skipped == 1 ? "y" : "ies",
                first_bad_key.layer_idx, first_bad_key.expert_idx,
                first_bad_gpu, cmd.cmd_seq, p.layer_idx);
        }
    }

    // ── Lock cached experts ────────────────────────────────────────────

    for (auto& er : state.experts) {
        if (er.is_arrived) {
            if (deps_.expert_cache->lock(er.key, er.target_gpu)) {
                er.is_locked = true;
            }
        }
    }

    // TD-EVICT-BOARD-DESYNC: the EvictScoreBoard now learns residency from the
    // ExpertCache's own add/evict choke-points (it is the cache's
    // ResidencyListener), so the former fetch-time loader_on_place feed is retired.
    // A fetch becomes board-resident only when the cache actually reserves its
    // stable slot (on_resident_added), and leaves the board on evict/cancel — so
    // the board tracks ACTUAL residency, never the fetch-request set (no stale
    // never-arrived entries). Recency for cached experts is re-stamped in
    // route_moe_by_loader (touch_existing).

    // ── Evict to make room for the missing experts (TD-FAR-EVICT) ──────
    // The expert cache reserve() does NOT auto-evict: it returns null when the
    // per-GPU stable zone is full, and the ELM then drops the interest (the
    // expert never arrives). Unlike RUN_MOE — whose orchestrator (reconcile_gpu_lru)
    // evicts the LRU victim before each prefetch — the progressive FETCH path had
    // no eviction step, so once the stable zone filled (after a few MoE layers the
    // VRAM-resident set exceeds capacity; this fixture is NOT VRAM-full-fit, hit
    // rate ~0.35) every subsequent routed expert failed to reserve and the MoE
    // finalized on a partial set, diverging from RUN_MOE. Free exactly enough
    // unlocked, not-needed-this-layer stable residents per target GPU so the
    // routed K can be admitted. Victims are picked from the current resident set
    // (any unlocked non-needed expert); they are unlocked across commands, so this
    // mirrors RUN_MOE's make-room-then-fetch without tracking a separate LRU.
    // TD-FAR-SLOT-RESERVE-STALL: when the batched inline-evict path owns make-room
    // (decode, num_seqs==1), skip this one-shot bulk evict entirely — ensure_residents
    // evicts a cheapest non-needed victim per reserve, so nothing under-frees/parks.
    const bool inline_evict_owns_makeroom =
        far_ensure_residents_enabled() && state.num_seqs == 1 &&
        evict_board_ && deps_.cuda_kernels_enabled;
    if (deps_.elm && deps_.expert_cache && !inline_evict_owns_makeroom) {
        // TD-FAR-EVICT / TD-FAR-EVICT-REROUTE: victim selection lives in the
        // SHARED gpu_loader::apply_far_evictions (core/gpu_loader/far_evict.cpp)
        // so the daemon and the CPU offline-sim drive the identical eviction code
        // over the identical engine classes. Build the index-aligned FarEvictFetch
        // view (state.experts + the sideband orchestrator victim map), then call
        // the shared helper. The A/B kill-switch (LS_EVICT_LRU_FALLBACK=0 → legacy
        // hash-order fallback) is unchanged; default ON.
        static const bool lru_fallback = [] {
            const char* v = std::getenv("LS_EVICT_LRU_FALLBACK");
            return !(v && v[0] == '0');
        }();
        const ipc::ExpertEvictionEntry* evicts =
            p.have_evict_map
                ? reinterpret_cast<const ipc::ExpertEvictionEntry*>(
                      deps_.sideband_base + ipc::IpcLayout::kExpertEvictionOff)
                : nullptr;
        std::vector<gpu_loader::FarEvictFetch> fetches(state.experts.size());
        for (size_t i = 0; i < state.experts.size(); ++i) {
            const auto& er = state.experts[i];
            auto& f = fetches[i];
            f.key = er.key;
            f.target_gpu = er.target_gpu;
            // Transient (zone=1) fetches reserve in the STREAMING zone — they
            // must not count toward stable make-room (union-aware cache
            // partitioning: the stable working set is never evicted for a
            // transient union fetch). zone==0 entries are byte-identical.
            f.is_real_fetch = !er.is_arrived && er.fetch_requested
                              && er.zone == 0;
            if (evicts && evicts[i].expert_idx != 0xFFFF) {
                f.has_victim = true;
                f.victim = make_key(evicts[i].layer_idx, evicts[i].expert_idx);
            }
        }
        gpu_loader::FarEvictStats st;
        gpu_loader::apply_far_evictions(
            *deps_.expert_cache, evict_board_ ? &*evict_board_ : nullptr,
            p.have_evict_map, lru_fallback,
            std::span<const gpu_loader::FarEvictFetch>(fetches), state.layer_idx,
            [this](memory::ExpertKey k, int g) {
                return deps_.elm->request_evict(k, g);
            },
            st);
        far_evict_honored_  += st.honored;
        far_evict_rejected_ += st.rejected;
        far_evict_fallback_ += st.fallback;
    }

    // ── Start H2D for selected missing experts ─────────────────────────
    // TD-PREFILL-FETCH-SEAM-SCALING: capacity-bounded WAVE issue. Fetches are
    // issued only up to each target GPU's free stable-slot count — an over-
    // capacity routed union (B>1 prefill chunks: ~125 experts/GPU vs ~60 free
    // slots) previously over-issued ALL of them: the un-reservable remainder
    // livelocked (re-issue → reserve fail → drop, millions of failed
    // completions/layer) while any_fetch_in_flight (host always warm under a
    // prepacked source) held the command open until the FULL per-command
    // deadline — 120 s per MoE layer with zero PCIe/NVMe traffic. Waves make
    // the fetch bandwidth-bound: issue what fits, compute+accumulate, evict,
    // issue the next wave (advance_progressive_moe drives the rolling passes).

    std::fill(moe_wave_accum_used_.begin(), moe_wave_accum_used_.end(), 0);
    if (moe_big_xray_enabled()) {
        state.xray_enter_ns = xray_now_ns();
        state.xray_last_ns = state.xray_enter_ns;
    }
    // C-6 early-kick η: the fetch window opens at issue_moe_wave (the missing-
    // expert H2D launch). Stamped for every forced-CPU layer so the fold can
    // report host-FFN-∥-fetch overlap for BOTH the early and late kick arms.
    if (cpu_layer_has_forced(state.layer_idx))
        cpu_fetch_win_start_ns_ = xray_now_ns();
    issue_moe_wave(state);

    // ── C-6 early-kick: kick the host CPU-expert FFN CONCURRENT with the fetch ──
    // The missing-expert H2Ds are now in flight on the copy streams (issue_moe_
    // wave above). route_moe_by_loader (the I8 solve) has already resolved this
    // layer's forced-CPU set. Kick the host FFN NOW so it overlaps the (big)
    // fetch window — the exact analogue of the resident-expert overlap pass — via
    // a light rank0 norm PRIME (produces normalized_hidden + records the input-
    // ready event; the routed top-K is already in scratch from attention
    // emit_gating) followed by start_cpu_forced_experts. The finalize's fold joins
    // this worker; its Phase-1 late kick is skipped (cpu_early_kick_layer_). No-op
    // unless early-kick + overlap are on, the layer has forced experts, and this is
    // the DCP(TP)>1 EP path (dcp_size<=1 folds inside dispatch_moe_internal, not
    // covered here). Kicked ONLY on the fetch-issuing call (num_seqs>=1), once.
    if (cpu_expert_early_kick_enabled() && cpu_expert_overlap_enabled()
        && cpu_layer_has_forced(state.layer_idx)
        && deps_.cuda_kernels_enabled && deps_.dcp_executor
        && deps_.dcp_executor->dcp_size() > 1
        && static_cast<int>(state.num_seqs) <= moe_chunk_capacity_) {
        const auto& tp_gpus = deps_.dcp_executor->gpus();
        std::vector<int> positions;
        positions.reserve(tp_gpus.size());
        for (const auto& g : tp_gpus)
            if (g.position >= 0) positions.push_back(g.position);
        if (!positions.empty()) {
            bool perslot = false, bf16 = false;
            predict_cpu_fold_format(state.layer_idx,
                                    static_cast<int>(state.num_seqs), perslot, bf16);
            InternalMoeParams pmp{};
            pmp.layer_idx = state.layer_idx;
            pmp.num_seqs  = state.num_seqs;
            pmp.gpu_idx   = static_cast<uint32_t>(positions[0]);
            pmp.moe_mode  = state.moe_mode;
            pmp.use_precomputed_gating = true;   // top-K already in scratch
            pmp.prime_cpu_input_only   = true;   // norm + input event, then return
            if (dispatch_moe_internal(pmp)) {
                start_cpu_forced_experts(state.layer_idx,
                                         static_cast<int>(state.num_seqs),
                                         positions, perslot, bf16);
                cpu_early_kick_layer_ = state.layer_idx;
                cpu_kick_perslot_     = perslot;
                cpu_kick_bf16_        = bf16;
            }
        }
    }

    // ── Set timeout deadline ───────────────────────────────────────────

    if (p.timeout_us > 0) {
        auto now = std::chrono::steady_clock::now();
        state.deadline_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count())
            + static_cast<uint64_t>(p.timeout_us) * 1000ULL;
    }

    // ── No incremental compute (TD-FAR-MULTICOMMIT) ────────────────────
    // The progressive path issues the up-front cross-GPU fetches above to
    // overlap H2D with the wait, but does NOT run any incremental MoE compute
    // here or per-arrival. Each compute pass rebuilds the bitset from the FULL
    // arrived-so-far residency and OVERWRITES moe_output, so the incremental
    // passes save no work — the single kFull finalize pass recomputes the entire
    // routed set anyway. Running them would also be a correctness hazard: each
    // pass that reached Phase 3 would add the MoE residual to the hidden state
    // again. So we compute exactly once, at finalize/quiescence, committing the
    // residual exactly once per layer (matching the validated RUN_MOE path).

    // ── TD-FAR-STREAM-GATE probe (env LS_FAR_STREAM_GATE, default OFF) ──
    // Device-side gate of the fetch→finalize handoff for the all-will-arrive
    // DECODE case: when it engages, every just-issued fetch is already
    // stream-ordered on its GPU's h2d stream, the finalize kExpertFfn streams
    // now wait on per-GPU h2d barrier events, and the gated experts have been
    // marked arrived — so arrived+skipped==total below and the command
    // finalizes IMMEDIATELY (single kFinalize enqueue; the GPU starts the
    // GEMMs the instant the copies land, no host per-expert detection).
    // Returns false in every other situation — flag off, B>1, wave split,
    // anything not yet on-stream — leaving the host-poll path byte-identical.
    const bool stream_gated = try_far_stream_gate(state);
    (void)stream_gated;  // consumed via arrived_count below

    // ── Determine phase ────────────────────────────────────────────────

    if (state.arrived_count + state.skipped_count + state.cpu_forced_count
            >= state.total_experts) {
        // All experts either resident, deliberately skipped by the decider, or
        // forced onto the host CPU device (C-6) — nothing to wait on; finalize
        // immediately with the resident subset (CPU experts fold at finalize).
        // (Also the TD-FAR-STREAM-GATE gated path: the barrier events make
        // this finalize's kExpertFfn work wait for the H2D copies device-side.)
        state.phase = ProgressiveMoePhase::kFinalize;
    } else {
        state.phase = ProgressiveMoePhase::kWaitingExperts;
    }

    // ── INV-MOE-OVERLAP: decode fetch-overlap split ─────────────────────
    // The missing-expert H2Ds are in flight (issued above, DMA on the copy
    // streams); enqueue the resident experts' wave-partial pass NOW so their
    // GEMMs run concurrently with the fetch instead of gated behind it. The
    // finalize then computes only the just-fetched remainder (kFinal).
    if (state.phase == ProgressiveMoePhase::kWaitingExperts
        && !state.big && state.num_seqs == 1
        && deps_.cuda_kernels_enabled && moe_resident_overlap_enabled()) {
        run_moe_overlap_pass(state);
    }

    // ── LS_FAR_GATED_FINAL: gated-final hybrid ──────────────────────────
    // The resident-overlap kPartial kernels are now enqueued on the kExpertFfn
    // streams (they execute DURING the in-flight DMA). Device-gate the fetched
    // remainder: arm one h2d-barrier per involved GPU, make its kExpertFfn
    // stream wait on it, mark the fetched experts arrived, and finalize
    // IMMEDIATELY — the kFinal replay + collectives queue BEHIND the barrier,
    // so the host's per-expert arrival detection AND the finalize host
    // dispatch run during the DMA window instead of after it. Compute is
    // identical to the host-poll path (same kPartial/kFinal passes, same
    // kernels, same order); only the enqueue time changes. Any commit
    // precondition failure → false → byte-identical host-poll path continues.
    if (state.phase == ProgressiveMoePhase::kWaitingExperts
        && !state.big && state.num_seqs == 1
        && deps_.cuda_kernels_enabled && moe_gated_final_enabled()
        && far_stream_gate_commit(state, "LS_FAR_GATED_FINAL")) {
        state.phase = ProgressiveMoePhase::kFinalize;
    }

    perf_trace::record(perf_trace::kMoeFetchIssued,
                       static_cast<uint16_t>(state.gpu_idx), state.cmd_seq,
                       state.layer_idx, 0);

    progressive_moe_ = std::move(state);

    // If already finalizing, run advance immediately.
    if (progressive_moe_->phase == ProgressiveMoePhase::kFinalize) {
        advance_progressive_moe();
    }

    // ── ExpertStats recency feed (env-gated; see dispatch_loader.cpp) ──
    // Routing for this layer is committed; feed the routed top-K so recency
    // self-advances. Daemon-thread cost — only run when the loader consumes it
    // (loader_shadow_), else it costs ~18 ms/token (REGRESSION_HUNT_5p2.md).
    feed_expert_stats(cmd, entries, have_weights);
}

// TD-far: is any requested-but-not-yet-arrived expert still genuinely making
// progress toward residency? "In flight" = the ELM/cache still has an active
// operation that can complete it: a VRAM slot reserved, an H2D DMA running, or
// a host (NVMe→RAM) read pending. Experts the F-6 decider skipped, or that are
// stuck ABSENT (no slot, nothing chasing them), are NOT in flight — waiting on
// them is pointless. Returns false when nothing can still arrive, so the caller
// finalizes immediately instead of burning the 50 ms deadline.
bool CommandDispatcher::any_fetch_in_flight(
        const ProgressiveMoeState& st) const {
    // TD-PREFILL-FETCH-SEAM-SCALING: only ISSUED fetches count — un-issued
    // experts of later waves are handled by the wave scheduler and must not
    // hold the command on the deadline.
    if (!deps_.elm) {
        // No lifecycle manager → we cannot distinguish in-flight from stuck.
        // Conservatively report "in flight" so the deadline path still governs.
        for (const auto& er : st.experts)
            if (er.issued && er.fetch_requested && !er.is_arrived) return true;
        return false;
    }
    for (const auto& er : st.experts) {
        if (er.is_arrived || !er.fetch_requested || !er.issued) continue;
        auto es = deps_.elm->state(er.key, er.target_gpu);
        const bool gpu_in_flight =
            es.gpu_tier == GpuTier::kReserved ||
            es.gpu_tier == GpuTier::kTransferring;
        const bool host_in_flight = es.host_tier == HostTier::kLoadingToRam;
        // TD-FAR-GATING: a requested expert whose host copy is WARM (mmap /
        // prepacked / host-RAM) is always reachable — the ELM can still reserve a
        // VRAM slot and run the H2D. Its gpu_tier is transiently kAbsent between
        // daemon poll cycles (ensure_resident registered the interest, but the
        // async arena-load / H2D chain — re-driven by process_arena_loads each
        // poll — has not yet reached RESERVED/TRANSFERRING this cycle) and the ELM
        // interest may have been momentarily reverted by a slot-pressure failure
        // that the next poll retries. Treating that window as "not in flight" let
        // advance_progressive_moe finalize the MoE BEFORE this token's routed
        // experts arrived — computing a partial (often empty) routed set and
        // diverging from the validated RUN_MOE path, which blocks on PREFETCH_BATCH
        // (CMP_ELM_EXPERT_READY) until every routed expert is resident. Honoring
        // host-warm reachability + ELM interest makes FETCH wait for the routed K
        // exactly like RUN_MOE (the per-command deadline remains the hard bound).
        // Only a genuinely UNREACHABLE expert (host cold with no loader AND not on
        // the GPU) reports not-in-flight, preserving the keeper's full-N-list
        // early-finalize (those padding entries never become resident).
        const bool elm_interested = es.interest_count > 0;
        const bool host_reachable =
            es.host_tier == HostTier::kWarm && es.gpu_tier != GpuTier::kDraining;
        if (gpu_in_flight || host_in_flight || elm_interested || host_reachable)
            return true;
    }
    return false;
}

// ── TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave helpers ─────────────────────

int CommandDispatcher::issue_moe_wave(ProgressiveMoeState& st) {
    // LS_MOE_BIG_XRAY shim: accumulate the host cost + issue count of every
    // wave issue into the command's x-ray, then defer to the real logic.
    if (!moe_big_xray_enabled()) return issue_moe_wave_inner(st);
    const uint64_t t0 = xray_now_ns();
    const int n = issue_moe_wave_inner(st);
    st.xray_issue_ns += xray_now_ns() - t0;
    if (n > 0) st.xray_fetches += static_cast<uint32_t>(n);
    return n;
}

int CommandDispatcher::issue_moe_wave_inner(ProgressiveMoeState& st) {
    if (!deps_.elm) return 0;
    // M3b: prefill-class waves' fetches are excluded from the online-
    // placement frequency signal (freq_table.py semantics: decode + chunk
    // counted, prefill excluded — its differently-skewed hot set pollutes
    // the EMA and the online layout plateaus below the decode optimum).
    // Discriminator: the BIG command flag OR a routed-union size beyond any
    // decode/chunk-verify wave (decode ~8-30 experts, chunk R≤16 ≤~64;
    // prefill chunks route essentially the whole expert set). Cheap: one
    // bool store per wave issue.
    deps_.elm->set_fetch_count_enabled(!st.big && st.experts.size() <= 96);
    // TD-FAR-SLOT-RESERVE-STALL: decode uses the batched inline-evict path (one
    // ensure_residents per GPU, cheapest-non-needed victim evicted per reserve).
    if (far_ensure_residents_enabled() && st.num_seqs == 1 &&
        evict_board_ && deps_.expert_cache && deps_.cuda_kernels_enabled) {
        return issue_moe_wave_ensure_residents(st);
    }
    // Budget fetches per target GPU by the free stable-zone slots so the ELM is
    // never asked to reserve past capacity (the old over-issue livelocked:
    // reserve fail → drop → per-cycle re-issue, deadline-bound). Null-backend /
    // no-cache test paths keep the legacy unbudgeted issue-all.
    const bool budgeted = deps_.cuda_kernels_enabled && deps_.expert_cache;
    // TD-PREFILL-MOE-BIG double-buffered waves (spec §4 Way 2): when a BIG
    // command's un-issued remainder exceeds a GPU's free stable slots (a
    // multi-wave layer), RESERVE a small slot budget out of this wave so the
    // NEXT wave's H2D transfers can be issued (run_moe_wave_pass issues them
    // the moment this wave's chunk-GGEMMs are enqueued) and stream while the
    // GPU computes. The reserve is BOUNDED (min(free/2, 8)): every extra wave
    // costs a full-batch grouped-GEMM sweep (absent experts still run the
    // zero-weight GEMM over all permuted rows), so an unbounded half split
    // DOUBLED the wave count for a ~36 ms/wave transfer win on the
    // arena-warm path (measured 6.2 s vs 1.9 s per MoE layer). Eight slots
    // stream ~220 MB of wave i+1 under wave i's compute — the useful overlap
    // on the cold/NVMe-bound path — at <= +15% wave count.
    constexpr int kDoubleBufferReserveSlots = 8;
    std::unordered_map<int, int> pending_per_gpu;
    if (st.big && budgeted) {
        for (const auto& er : st.experts) {
            if (er.is_arrived || er.issued || !er.fetch_requested) continue;
            if (er.zone == 0) ++pending_per_gpu[er.target_gpu];
        }
    }
    std::unordered_map<int, int> budget;
    // Transient (zone=1) fetches: budget by free STREAMING slots so the ELM is
    // never handed a guaranteed-to-fail streaming reserve; an exhausted
    // streaming budget flips the entry to zone=0 (today's stable demand path,
    // which then takes the stable budget below). zone==0-only commands never
    // touch this map (byte-identical).
    std::unordered_map<int, int> stream_budget;
    int issued = 0;
    for (auto& er : st.experts) {
        if (er.is_arrived || er.issued || !er.fetch_requested) continue;
        if (budgeted && er.zone != 0) {
            auto sit = stream_budget.find(er.target_gpu);
            if (sit == stream_budget.end())
                sit = stream_budget.emplace(
                    er.target_gpu,
                    deps_.expert_cache->free_slots(
                        er.target_gpu, memory::CacheZone::kStreaming)).first;
            if (sit->second > 0) --sit->second;
            else er.zone = 0;  // overflow → stable fallback
        }
        if (budgeted && er.zone == 0) {
            auto it = budget.find(er.target_gpu);
            if (it == budget.end()) {
                int free = deps_.expert_cache->free_slots(
                    er.target_gpu, memory::CacheZone::kStable);
                // TD-ORCH-ELM-COMPLETION-LIVELOCK: the double-buffer reserve
                // applies only when the GPU has ANY free stable slot. The old
                // std::max(1, ...) FABRICATED a budget of 1 on a zero-free
                // GPU, issuing one guaranteed-to-fail ensure_resident whose
                // reserve-fail dropped the interest — and the kWaitingExperts
                // re-drive then hammered that one entry every daemon cycle
                // (~345k failed lifecycle completions/s for one key).
                if (st.big && free > 0) {
                    auto pit = pending_per_gpu.find(er.target_gpu);
                    if (pit != pending_per_gpu.end() && pit->second > free) {
                        const int reserve =
                            std::min(free / 2, kDoubleBufferReserveSlots);
                        free = std::max(1, free - reserve);
                    }
                }
                it = budget.emplace(er.target_gpu, free).first;
            }
            if (it->second <= 0) continue;  // this GPU's wave is full
            --it->second;
        }
        deps_.elm->ensure_resident(
            er.key, er.target_gpu, to_zone(er.zone),
            st.cmd_seq, /*priority=*/kDemandFetchPriority, /*delay_us=*/0);
        er.issued = true;
        ++issued;
    }
    return issued;
}

// ── TD-FAR-STREAM-GATE: device-side fetch→finalize gate (flag-gated probe) ──
//
// WHY: keeper52's MoE decode wall is host-completion-DETECTION-bound, not
// bandwidth-bound (spec/TECH_DEBT.md TD-FAR-STREAM-GATE). Both GPUs memcpy at
// ~44 GB/s (632/635 µs per fetch), but the single daemon thread detects each
// H2D completion at 991 µs/expert mean (355 µs median — bimodal: the thread is
// off doing attention/compute and simply not polling during the gaps), and the
// finalize cannot even be ENQUEUED until every per-expert `is_arrived` poll has
// fired. Host detect+queue ≈ 957 µs × ~600 fetches ≈ 574 ms/tok — the entire
// daemon MoE wall — while the actual parallel DMA (~205 ms/tok) hides under it.
//
// WHAT (the low-hanging HALF): keep the progressive machine, but replace the
// per-expert host arrival polls with ONE device-side edge. After issue_moe_wave
// has pushed every routed fetch of the layer onto the transfer engine and
// flush_staged() has put all of them (≤ max_inflight_per_gpu = 8) onto each
// GPU's h2d stream, record a BARRIER EVENT per involved GPU at the tail of its
// h2d stream — the "streamWaitEvent-my-no-op" doorbell marker: it carries no
// payload, it simply fires when every copy enqueued before it has landed — and
// make the same GPU's kExpertFfn stream wait on it (cudaStreamWaitEvent
// semantics via DeviceBackend, INV-GPU-1; portable to hipStreamWaitEvent). The
// finalize GEMMs are then enqueued IMMEDIATELY (the reserve step already
// assigned every expert's VRAM slot at issue time — entry->vram_address and
// the projection offsets are set by ExpertCache::reserve(), so the finalize
// pointer plan is fully known before the bytes arrive; the ready-bit is a
// host-bookkeeping detail the pointer build never reads). The host then syncs
// exactly ONCE per layer — the existing PendingCompute kExpertFfn event that
// defers CMP_COMPUTE_DONE (TD-FAR-ASYNC) — instead of ~8 per-expert
// detections. The ELM/transfer-engine completion chain still runs in the
// background on later poll cycles (mark_ready, interest completion, arena slot
// release) — it is merely OFF the critical path; end-of-command cache state is
// identical to the host-poll path.
//
// WHY DECODE IS THE TRACTABLE CASE: at B==1 the routed union (top-K) always
// fits the free stable slots, so waves are kNone (prefill-only) — the machine
// already waits-for-ALL-then-finalizes-ONCE, i.e. there is no incremental
// overlap to preserve and a single all-arrive barrier per GPU is exactly
// equivalent in WHAT it computes (same experts, same slots, same GEMM order —
// only WHEN the enqueue happens changes, and the barrier guarantees execution
// still follows the copies).
//
// FALLBACK CONDITIONS (any → return false, host-poll path runs unchanged):
//   - flag off / B>1 / null backends (no CUDA, no stream manager, no ELM, no
//     transfer engine, no cache);
//   - some requested expert was NOT issued in the first wave (capacity split →
//     the rolling-wave machine must stay host-driven);
//   - any pending expert is not in GpuTier::kTransferring with its DMA
//     actually DISPATCHED on the h2d stream (i.e. genuinely cold host tier,
//     reserve failure, arena backpressure, dedup-against-staged, or > 8
//     copies per GPU) — the barrier can only fence stream-ordered copies, so
//     device-gating a partial/unreachable set would compute garbage; those
//     cases keep the deadline/partial host escape. Note the fallback is safe
//     even if it triggers after some stream-waits were already enqueued: a
//     stray kExpertFfn wait on an h2d barrier only delays compute behind
//     copies that the host path waits out anyway (a copy, once dispatched,
//     always completes and fires its event).
//
// The FULL fix — enqueue the GEMMs fully up front, remove the progressive
// machine for decode, hybrid deadline/partial handling, evict-after-compute
// via device events — is tracked in TD-FAR-STREAM-GATE.
bool CommandDispatcher::try_far_stream_gate(ProgressiveMoeState& st) {
    // Env flag, read once (same pattern as LS_LOADER_ACT / LS_EVICT_LRU_FALLBACK).
    // Default OFF → this function is a no-op and the legacy path is byte-identical.
    static const bool enabled = [] {
        const char* v = std::getenv("LS_FAR_STREAM_GATE");
        return v && v[0] && v[0] != '0';
    }();
    if (!enabled) return false;
    return far_stream_gate_commit(st, "LS_FAR_STREAM_GATE");
}

// LS_FAR_GATED_FINAL: resident-overlap + device-gated finalize hybrid (see
// header). DEFAULT ON (user decision 2026-07-21): B=1 wall-neutral measured
// (bit-identical trajectory + goldens), removes the daemon's ~30 ms/tok
// arrival-detection spin from every fetch wave — host capacity for
// multi-request serving. Fail-safe: non-gateable waves take the host path
// (per-reason fb_* counters, teardown engagement line). =0 opts out.
bool CommandDispatcher::moe_gated_final_enabled() {
    if (moe_gated_final_enabled_ < 0) {
        const char* e = std::getenv("LS_FAR_GATED_FINAL");
        moe_gated_final_enabled_ = (e && e[0] && e[0] == '0') ? 0 : 1;
    }
    return moe_gated_final_enabled_ == 1;
}

bool CommandDispatcher::far_stream_gate_commit(ProgressiveMoeState& st,
                                               const char* tag) {
    // DECODE only (B==1). Prefill / batched paths keep the host-driven
    // progressive machine (waves + deadline/partial escapes) untouched.
    if (st.num_seqs != 1) return false;
    if (!deps_.cuda_kernels_enabled || !deps_.stream_manager
        || !deps_.transfer_engine || !deps_.elm || !deps_.expert_cache) {
        ++gated_final_fb_other_;
        return false;
    }

    // Collect the pending fetches. ALL requested experts must have been issued
    // by the first issue_moe_wave — an un-issued remainder means the routed
    // union exceeded free stable capacity and the rolling-wave machine (waves
    // != kNone) will drive this command: host path only.
    std::vector<size_t> pending;
    pending.reserve(st.experts.size());
    for (size_t i = 0; i < st.experts.size(); ++i) {
        const auto& er = st.experts[i];
        if (er.is_arrived || !er.fetch_requested) continue;
        if (!er.issued) {  // wave split → host-driven
            ++gated_final_fb_unissued_;
            return false;
        }
        pending.push_back(i);
    }
    // Nothing in flight (fully resident / all skipped): the existing immediate-
    // finalize branch already handles this with zero waiting — nothing to gate.
    if (pending.empty()) return false;

    // Push staged copies onto the h2d streams NOW. enqueue_h2d dispatches only
    // the first min_dispatch_per_gpu (2) inline and stages the rest; the daemon
    // loop would flush next cycle anyway — doing it here (same thread, same
    // call the loop makes) just moves the flush before the barrier record.
    deps_.transfer_engine->flush_staged();

    // Verify every pending expert is genuinely stream-ordered: ELM chain
    // reached kTransferring (slot reserved + H2D handed to the transfer
    // engine), the DMA is DISPATCHED (not still staged — e.g. >8 copies on one
    // GPU or competing prefetch traffic), and the finalize pointer plan exists
    // (reserve() set vram_address + projection offsets). Any miss → host path.
    for (size_t i : pending) {
        const auto& er = st.experts[i];
        const auto es = deps_.elm->state(er.key, er.target_gpu);
        if (es.gpu_tier != GpuTier::kTransferring) {
            ++gated_final_fb_not_transferring_;
            return false;
        }
        if (!deps_.transfer_engine->is_dispatched_h2d(er.key, er.target_gpu)) {
            ++gated_final_fb_not_dispatched_;
            return false;
        }
        const auto* ce = deps_.expert_cache->lookup(er.key, er.target_gpu);
        if (!ce || !ce->vram_address) {
            ++gated_final_fb_other_;
            return false;
        }
        if (er.target_gpu < 0
            || static_cast<size_t>(er.target_gpu) >= deps_.device_backends.size()
            || !deps_.device_backends[er.target_gpu]) {
            ++gated_final_fb_other_;
            return false;
        }
    }

    // Involved GPUs (tiny set — EP split of the routed top-K).
    std::vector<int> gpus;
    for (size_t i : pending) {
        const int g = st.experts[i].target_gpu;
        if (std::find(gpus.begin(), gpus.end(), g) == gpus.end())
            gpus.push_back(g);
    }

    // The device-side edge: one barrier event per involved GPU on its h2d
    // stream; that GPU's kExpertFfn stream (where the finalize permute/GEMM/
    // unpermute — eager or CUDA-graph replay — and the EP allreduce enqueue)
    // waits on it. The event is destroyed immediately after the wait is
    // enqueued (dependency is retained by the driver — same pattern as
    // setup_spec_pipeline's embed handoff).
    for (const int g : gpus) {
        void* barrier = deps_.transfer_engine->record_h2d_barrier(g);
        if (!barrier) {  // see fallback-safety note above
            ++gated_final_fb_other_;
            return false;
        }
        void* ffn = deps_.stream_manager->stream(g, compute::StreamId::kExpertFfn);
        deps_.device_backends[g]->stream_wait_event(ffn, barrier);
        deps_.device_backends[g]->destroy_event(barrier);
    }
    ++gated_final_engaged_;

    // Commit: mark the gated experts arrived (the finalize bitset is built
    // from er.is_arrived) and take the eviction lock, exactly like the host
    // arrival poll in advance_progressive_moe would. lock() works on the
    // reserved-not-yet-ready entry (keyed on residents map presence), and
    // nothing can evict a kTransferring entry between here and the finalize —
    // the whole sequence is synchronous within this command dispatch.
    for (size_t i : pending) {
        auto& er = st.experts[i];
        er.is_arrived = true;
        ++st.arrived_count;
        if (deps_.expert_cache->lock(er.key, er.target_gpu))
            er.is_locked = true;
        // perf_trace: under the gate this stamp is the host COMMIT time (copy
        // stream-ordered + barrier armed), not a completion-detection time —
        // the detection artifact this probe removes (TD-FAR-STREAM-GATE says
        // to relabel fetch_dead as a detection-order metric for this reason).
        perf_trace::record(perf_trace::kExpertArrived,
                           static_cast<uint16_t>(er.target_gpu), st.cmd_seq,
                           (static_cast<uint32_t>(er.key.layer_idx) << 16) |
                               er.key.expert_idx,
                           0);
    }

    // One-time proof the gate engaged (grep target for the A/B harness).
    static bool logged = false;
    if (!logged) {
        logged = true;
        spdlog::info("FAR stream-gate ACTIVE ({}): decode MoE "
                     "finalize is device-gated on per-GPU h2d barrier events "
                     "(first layer {}: {} fetches across {} GPU(s))",
                     tag, st.layer_idx, pending.size(), gpus.size());
    }
    return true;
}

// TD-FAR-SLOT-RESERVE-STALL: batched inline-evict residency for decode. One
// ensure_residents call per target GPU with a cheapest-first, pre-filtered victim
// list from the evict board — no free-slot budget/skip, no reserve-fail park.
// Replaces the legacy per-expert ensure_resident + one-shot apply_far_evictions.
int CommandDispatcher::issue_moe_wave_ensure_residents(ProgressiveMoeState& st) {
    // LS_LOADER_MACH_PROF: victim-gather vs ELM-call split (default-off branch).
    const bool prof = mach_prof_.enabled;
    auto prof_now = [] {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    uint64_t prof_t0 = prof ? prof_now() : 0;
    uint64_t prof_vict_ns = 0, prof_elm_ns = 0;
    auto pack = [](memory::ExpertKey k) -> uint64_t {
        return (static_cast<uint64_t>(k.layer_idx) << 16) |
               static_cast<uint16_t>(k.expert_idx);
    };
    // Per-GPU missing-expert reqs + the full needed-key set (this command's routed
    // experts on that GPU — protects the batch's own reserves from being evicted).
    std::unordered_map<int, std::vector<ExpertLifecycleManager::ResidentReq>> by_gpu;
    std::unordered_map<int, std::unordered_set<uint64_t>> needed;
    for (const auto& er : st.experts) {
        if (!er.fetch_requested) continue;
        needed[er.target_gpu].insert(pack(er.key));
        if (er.is_arrived || er.issued) continue;
        by_gpu[er.target_gpu].push_back({er.key, to_zone(er.zone)});
    }
    for (auto& [g, reqs] : by_gpu) {
        // Cheapest stable victims, pre-filtered: not needed this command, not
        // locked. request_evict (inside ensure_residents) additionally skips any
        // still-un-evictable (kTransferring / interests). A generous prefix; widen
        // to full residency only if the cheap prefix is entirely guarded out.
        std::vector<memory::ExpertKey> victims;
        int resident = 0;
        const auto& keep = needed[g];
        auto guarded = [&](const memory::ExpertKey& vk) {
            return keep.count(pack(vk)) || deps_.expert_cache->is_locked(vk, g);
        };
        const int want = static_cast<int>(reqs.size()) * 2 + 16;
        evict_board_->cheapest_keys(g, want, victims, &resident);
        victims.erase(std::remove_if(victims.begin(), victims.end(), guarded),
                      victims.end());
        if (static_cast<int>(victims.size()) < static_cast<int>(reqs.size()) &&
            want < resident) {
            evict_board_->cheapest_keys(g, resident, victims, nullptr);
            victims.erase(std::remove_if(victims.begin(), victims.end(), guarded),
                          victims.end());
        }
        if (prof) {
            const uint64_t t = prof_now();
            prof_vict_ns += t - prof_t0;
            prof_t0 = t;
        }
        deps_.elm->ensure_residents(g, reqs, victims, st.cmd_seq,
                                    kDemandFetchPriority);
        if (prof) {
            const uint64_t t = prof_now();
            prof_elm_ns += t - prof_t0;
            prof_t0 = t;
        }
    }
    // Mark issued ONLY for experts that secured a slot; a genuinely-absent expert
    // (no evictable victim — cannot happen in decode where need << capacity) stays
    // unissued so the wave machine re-issues it (never wedge issued=true with no
    // transfer — that half of the stall bug).
    int issued = 0;
    for (auto& er : st.experts) {
        if (er.is_arrived || er.issued || !er.fetch_requested) continue;
        if (deps_.elm->state(er.key, er.target_gpu).gpu_tier != GpuTier::kAbsent) {
            er.issued = true;
            ++issued;
        }
    }
    if (prof) {
        prof_elm_ns += prof_now() - prof_t0;
        mach_prof_.add(LoaderMachProf::kEnsureVictims, prof_vict_ns);
        mach_prof_.add(LoaderMachProf::kEnsureResidents, prof_elm_ns);
    }
    return issued;
}

bool CommandDispatcher::run_moe_wave_pass(ProgressiveMoeState& st) {
    if (!deps_.cuda_kernels_enabled || !deps_.live_config || !deps_.stream_manager)
        return false;
    const int n_experts = deps_.live_config->model.n_routed_experts;
    if (n_experts <= 0) return false;

    const bool xray = moe_big_xray_enabled();
    const uint64_t xray_t0 = xray ? xray_now_ns() : 0;

    // Group the arrived-and-uncomputed experts per target GPU.
    std::unordered_map<int, std::vector<size_t>> per_gpu;
    for (size_t i = 0; i < st.experts.size(); ++i) {
        const auto& er = st.experts[i];
        if (!er.is_arrived || er.computed) continue;
        if (er.target_gpu < 0
            || static_cast<size_t>(er.target_gpu) >= moe_scratch_.size())
            continue;
        // TD-MOE-EP-XTP-WAVES: expert-only (non-DCP) ranks never join wave
        // passes — kPartial needs the broadcast hidden/top-K which only the
        // finalize path provides. Their arrived experts stay un-computed and
        // are covered by the finalize's EP-XTP dispatch (they merely hold
        // their slots until then).
        if (std::find(ep_xtp_gpus_.begin(), ep_xtp_gpus_.end(),
                      er.target_gpu) != ep_xtp_gpus_.end())
            continue;
        per_gpu[er.target_gpu].push_back(i);
    }
    if (per_gpu.empty()) return false;

    // The wave accumulator must exist on every involved GPU (else waves are
    // impossible; the caller falls back to issue-or-degrade).
    for (const auto& [gpu, idxs] : per_gpu) {
        if (!moe_scratch_[gpu].moe_wave_accum || !expert_dev(gpu)
            || !moe_scratch_[gpu].topk_weights)
            return false;
    }

    for (auto& [gpu, idxs] : per_gpu) {
        auto& bs = moe_scratch_[gpu].expert_resident_bitset;
        std::memset(bs.data(), 0, bs.size());
        for (size_t idx : idxs) {
            auto& er = st.experts[idx];
            const int eidx = er.key.expert_idx;
            bs[eidx / 8] |= static_cast<uint8_t>(1u << (eidx % 8));
            er.computed = true;
            st.wave_batch.push_back(idx);
        }

        InternalMoeParams mp{};
        mp.layer_idx = st.layer_idx;
        mp.num_seqs  = st.num_seqs;
        mp.gpu_idx   = static_cast<uint32_t>(gpu);
        mp.moe_mode  = st.moe_mode;
        // kPartial returns right after Step 5 + accumulate — no unpermute,
        // shared expert, allreduce, residual or commit — so no cross-rank
        // coordination is needed: dispatch each rank directly.
        mp.phase = MoeDispatchPhase::kPreAllreduce;
        mp.bitset_precomputed = true;
        mp.use_precomputed_gating = true;
        mp.wave_pass = MoeWavePass::kPartial;
        mp.chunk_tokens = st.chunk_tokens;  // TD-PREFILL-MOE-BIG

        // LS_MOE_BIG_XRAY: bracket this wave's kExpertFfn enqueue with a GPU
        // timing-event pair; reaped at wave drain (the wave event below is
        // recorded after t_end on the same stream, so both have fired).
        void* xray_t_start = nullptr;
        compute::DeviceBackend* xray_be = nullptr;
        if (xray && static_cast<size_t>(gpu) < deps_.device_backends.size()
            && deps_.device_backends[gpu]) {
            xray_be = deps_.device_backends[gpu];
            xray_t_start = xray_be->create_timing_event();
            if (xray_t_start)
                xray_be->record_event(xray_t_start, deps_.stream_manager->stream(
                    gpu, compute::StreamId::kExpertFfn));
        }

        if (!dispatch_moe_internal(mp)) {
            spdlog::error("progressive MoE wave pass failed on gpu {} "
                          "(layer {})", gpu, st.layer_idx);
        }

        if (xray_t_start) {
            void* xray_t_end = xray_be->create_timing_event();
            if (xray_t_end) {
                xray_be->record_event(xray_t_end, deps_.stream_manager->stream(
                    gpu, compute::StreamId::kExpertFfn));
                st.wave_timing_events.emplace_back(gpu, xray_t_start, xray_t_end);
            } else {
                xray_be->destroy_event(xray_t_start);
            }
        }

        void* ev = create_and_record_event(gpu, compute::StreamId::kExpertFfn);
        if (ev) st.wave_events.emplace_back(gpu, ev);
    }

    st.wave_ran = true;
    if (xray) {
        st.xray_wave_enqueue_ns += xray_now_ns() - xray_t0;
        if (!st.wave_batch.empty()) ++st.xray_waves;
    }

    // TD-PREFILL-MOE-BIG Way 2 (double-buffered pipeline): this wave's
    // chunk-GGEMMs are now ENQUEUED (async on the kExpertFfn streams) — issue
    // the NEXT wave's H2D fetches immediately into the slot half reserved by
    // the issue budget split, so the transfer engine streams wave i+1 while
    // the GPU computes wave i. The drain step still gates EVICTION of this
    // wave's experts on the compute events (slots are only reused after the
    // GPU is done reading them — INV-FAR-WAVE safety is unchanged).
    if (st.big && !st.wave_batch.empty())
        issue_moe_wave(st);

    return !st.wave_batch.empty();
}

// ── INV-MOE-OVERLAP: decode fetch-overlap split ─────────────────────────────
// Enqueue a wave-partial pass over the experts ALREADY resident at fetch-issue
// time so their compute overlaps the in-flight missing-expert H2Ds. Unlike the
// prefill rolling waves: (a) the EP-XTP extra ranks participate (after the
// rank0 hidden/top-K broadcast — decode uses precomputed gating, so the
// routing is already in rank0's scratch, stream-ordered behind attention);
// (b) no wave_events / wave_batch — the computed experts stay locked and
// RESIDENT (no drain, no evict; decode wants them cached for future tokens).
// The finalize then runs kFinal over only the just-fetched remainder
// (arrived ∧ !computed) and unpermutes from the accumulator — bit-identical
// to the single-pass result (each permuted row is nonzero in exactly one
// pass; the excluded experts' rows come from the zeroed weight buffer).
bool CommandDispatcher::run_moe_overlap_pass(ProgressiveMoeState& st) {
    if (!deps_.cuda_kernels_enabled || !deps_.live_config
        || !deps_.stream_manager || !deps_.expert_cache)
        return false;
    if (st.num_seqs != 1 || st.big) return false;
    const int n_experts = deps_.live_config->model.n_routed_experts;
    if (n_experts <= 0) return false;

    // Group the arrived-and-uncomputed (i.e. initially-resident) experts per
    // target GPU — including extra (expert-only) ranks.
    std::unordered_map<int, std::vector<size_t>> per_gpu;
    for (size_t i = 0; i < st.experts.size(); ++i) {
        const auto& er = st.experts[i];
        if (!er.is_arrived || er.computed) continue;
        if (er.target_gpu < 0
            || static_cast<size_t>(er.target_gpu) >= moe_scratch_.size())
            continue;
        per_gpu[er.target_gpu].push_back(i);
    }
    if (per_gpu.empty()) return false;

    // The wave accumulator + routing scratch must exist on every involved GPU.
    for (const auto& [gpu, idxs] : per_gpu) {
        if (!moe_scratch_[gpu].moe_wave_accum || !expert_dev(gpu)
            || !moe_scratch_[gpu].topk_weights)
            return false;
    }

    // Broadcast rank0's normalized hidden + top-K to EVERY extra rank this
    // command touches (resident now OR being fetched): the finalize skips its
    // own broadcast when xtp_broadcast_done, so extras that become active
    // only at finalize (fetched experts) must be covered here too.
    if (!ep_xtp_gpus_.empty()) {
        InternalMoeParams mpb{};
        mpb.layer_idx = st.layer_idx;
        mpb.num_seqs  = st.num_seqs;
        mpb.gpu_idx   = st.gpu_idx;
        mpb.moe_mode  = st.moe_mode;
        mpb.use_precomputed_gating = true;
        std::vector<int> bcast;
        for (int g : ep_xtp_gpus_) {
            for (const auto& er : st.experts) {
                if (er.target_gpu == g
                    && (er.is_arrived || er.fetch_requested)) {
                    bcast.push_back(g);
                    break;
                }
            }
        }
        if (!bcast.empty()) {
            if (ep_xtp_broadcast(mpb, bcast, /*after_rank0_dispatch=*/false)) {
                st.xtp_broadcast_done = true;
            } else {
                // Extras cannot compute without the broadcast — drop them from
                // this pass; the finalize's own broadcast + dispatch covers
                // them (xtp_broadcast_done stays false).
                spdlog::warn("INV-MOE-OVERLAP: EP-XTP broadcast failed (layer "
                             "{}) — extras deferred to finalize", st.layer_idx);
                for (int g : bcast) per_gpu.erase(g);
                if (per_gpu.empty()) return false;
            }
        }
    }

    bool enqueued = false;
    for (auto& [gpu, idxs] : per_gpu) {
        auto& bs = moe_scratch_[gpu].expert_resident_bitset;
        std::memset(bs.data(), 0, bs.size());
        for (size_t idx : idxs) {
            auto& er = st.experts[idx];
            const int eidx = er.key.expert_idx;
            bs[eidx / 8] |= static_cast<uint8_t>(1u << (eidx % 8));
            // Marked computed BEFORE dispatch (same convention as
            // run_moe_wave_pass): a dispatch failure degrades (rows missing,
            // like a timed-out expert) rather than double-counting.
            er.computed = true;
        }

        InternalMoeParams mp{};
        mp.layer_idx = st.layer_idx;
        mp.num_seqs  = st.num_seqs;
        mp.gpu_idx   = static_cast<uint32_t>(gpu);
        mp.moe_mode  = st.moe_mode;
        // kPartial returns right after Step 5 + accumulate — no unpermute /
        // shared expert / allreduce / residual / commit, so each rank is
        // dispatched directly (no cross-rank coordination).
        mp.phase = MoeDispatchPhase::kPreAllreduce;
        mp.bitset_precomputed = true;
        mp.use_precomputed_gating = true;
        mp.wave_pass = MoeWavePass::kPartial;
        mp.chunk_tokens = 0;

        if (!dispatch_moe_internal(mp)) {
            spdlog::error("INV-MOE-OVERLAP: overlap pass failed on gpu {} "
                          "(layer {})", gpu, st.layer_idx);
            continue;
        }
        enqueued = true;
    }

    if (enqueued) st.wave_ran = true;
    return enqueued;
}

// Queued-command backpressure (burst-published FAR/FETCH sweeps): block-drain
// the ACTIVE progressive MoE to finalization before dispatching the next
// routed-MoE command. Semantically identical to serial pacing — layer L+1's
// attention consumes layer L's MoE output anyway, so waiting here just moves
// the wait from the orchestrator's ring turnaround into the daemon. Each
// iteration runs the SAME progress pair the daemon cycle runs: the lifecycle
// pump (transfer/NVMe → ELM → arrivals; the ONLY producer of
// ExpertCache::mark_all_ready) then advance_progressive_moe (waves, deadline
// burn, finalize + completion publish). The progressive machinery's own
// per-command deadlines bound the drain; the 150 s cap (above the 120 s
// prefill FETCH deadline) only guards against a wedged state machine.
bool CommandDispatcher::drain_progressive_moe(const char* ctx) {
    if (!progressive_moe_.has_value()) return true;
    if (!lifecycle_pump_) return false;  // no pump seam → legacy reject
    const auto cap = std::chrono::steady_clock::now()
                     + std::chrono::seconds(150);
    while (progressive_moe_.has_value()) {
        lifecycle_pump_();
        advance_progressive_moe();
        if (!progressive_moe_.has_value()) break;
        if (std::chrono::steady_clock::now() > cap) {
            spdlog::error("{}: progressive-MoE drain cap expired (layer {})",
                          ctx, progressive_moe_->layer_idx);
            return false;
        }
    }
    return true;
}

bool CommandDispatcher::advance_progressive_moe() {
    if (!progressive_moe_.has_value()) return false;

    auto& st = *progressive_moe_;

    // LS_MOE_BIG_XRAY: attribute the wall since the last advance call to the
    // phase in effect over that gap — kWaitingExperts = fetch-WAIT (H2D
    // arrival detection), kWaveDrain = wave GPU-drain wait. The daemon loop
    // calls advance every poll cycle while a progressive MoE is active, so
    // these gaps tile the command wall (minus the timed issue/enqueue spans).
    if (moe_big_xray_enabled() && st.xray_last_ns) {
        const uint64_t now = xray_now_ns();
        const uint64_t d = now - st.xray_last_ns;
        if (st.phase == ProgressiveMoePhase::kWaveDrain)
            st.xray_wave_drain_ns += d;
        else if (st.phase == ProgressiveMoePhase::kWaitingExperts)
            st.xray_wait_experts_ns += d;
        st.xray_last_ns = now;
    }

    // ── kWaveDrain: wait for the wave-partial pass GPU events ──────────
    // (TD-PREFILL-FETCH-SEAM-SCALING) The wave's kernels read the wave
    // experts' VRAM slots; only after every rank's kExpertFfn event fires may
    // those slots be evicted (freed for the next wave's H2Ds).
    if (st.phase == ProgressiveMoePhase::kWaveDrain) {
        bool all_ready = true;
        for (auto& [gpu, ev] : st.wave_events) {
            auto [status, err] = deps_.stream_manager->query_event(ev, gpu);
            if (status == compute::EventStatus::kNotReady) {
                all_ready = false;
                break;
            }
            if (status == compute::EventStatus::kError) {
                // Surface but proceed — the compute error will be reported by
                // downstream sync points; blocking here would wedge the daemon.
                spdlog::error("progressive MoE wave drain: event error {} on "
                              "gpu {} (layer {})", err, gpu, st.layer_idx);
            }
        }
        if (!all_ready) return false;

        for (auto& [gpu, ev] : st.wave_events)
            deps_.stream_manager->destroy_event(ev, gpu);
        st.wave_events.clear();

        // LS_MOE_BIG_XRAY: reap this wave's GPU timing pairs (both events
        // precede the drained wave events on the same streams — complete).
        for (auto& [gpu, t_start, t_end] : st.wave_timing_events) {
            if (static_cast<size_t>(gpu) < deps_.device_backends.size()
                && deps_.device_backends[gpu]) {
                auto* be = deps_.device_backends[gpu];
                const float ms = be->event_elapsed_ms(t_start, t_end);
                if (ms > 0.0f) st.xray_wave_gpu_ms += ms;
                be->destroy_event(t_start);
                be->destroy_event(t_end);
            }
        }
        st.wave_timing_events.clear();

        // GPU is done reading the wave's experts — unlock + evict them so the
        // next wave's fetches have slots.
        for (size_t idx : st.wave_batch) {
            auto& er = st.experts[idx];
            if (er.is_locked && deps_.expert_cache) {
                deps_.expert_cache->unlock(er.key, er.target_gpu);
                er.is_locked = false;
            }
            if (deps_.elm) {
                deps_.elm->request_evict(er.key, er.target_gpu);
            } else if (deps_.expert_cache) {
                deps_.expert_cache->evict(er.key, er.target_gpu);
            }
        }
        st.wave_batch.clear();

        issue_moe_wave(st);
        st.phase = ProgressiveMoePhase::kWaitingExperts;
        return false;
    }

    // ── kWaitingExperts: check for new arrivals ────────────────────────

    if (st.phase == ProgressiveMoePhase::kWaitingExperts) {
        // Poll expert cache for newly-ready experts.
        bool any_new = false;
        for (auto& er : st.experts) {
            if (er.is_arrived) continue;
            // F-6: experts the decider skipped were never fetched — don't await
            // them (they degrade gracefully like timed-out experts).
            if (!er.fetch_requested) continue;
            const auto* ce = deps_.expert_cache->lookup(er.key, er.target_gpu);
            if (ce && (ce->sub_components_ready & memory::SubComponent::kAll)
                        == memory::SubComponent::kAll) {
                er.is_arrived = true;
                // perf_trace: stamp this expert's arrival (ns set inside record()).
                // Per-GPU last-arrival per cmd → fetch wall + fastest/dead (x-ray).
                perf_trace::record(perf_trace::kExpertArrived,
                                   static_cast<uint16_t>(er.target_gpu), st.cmd_seq,
                                   (static_cast<uint32_t>(er.key.layer_idx) << 16) |
                                       er.key.expert_idx,
                                   0);
                ++st.arrived_count;
                // Lock the newly-arrived expert.
                if (deps_.expert_cache->lock(er.key, er.target_gpu)) {
                    er.is_locked = true;
                }
                any_new = true;
            }
        }

        // TD-FAR-GATING: re-issue ensure_resident for any requested expert the
        // ELM has DROPPED — gpu ABSENT with no interest, yet host data is reachable
        // (warm). A single up-front ensure_resident (handle_fetch_and_run_moe) can
        // fail transiently when the demand fetch contends for arena slots under the
        // FLT_MAX-priority burst (slot-pressure revert clears the interest). Unlike
        // RUN_MOE's blocking PREFETCH_BATCH — re-driven implicitly by the daemon
        // until the harness sees CMP_ELM_EXPERT_READY — the progressive path has no
        // retry, so a dropped expert stays absent forever and the MoE finalizes on
        // a partial routed set, diverging from RUN_MOE. Re-driving here makes the
        // routed K eventually resident exactly like RUN_MOE (bounded by deadline).
        if (deps_.elm) {
            for (auto& er : st.experts) {
                // TD-PREFILL-FETCH-SEAM-SCALING: only re-drive ISSUED fetches
                // (un-issued ones belong to later waves).
                if (er.is_arrived || !er.fetch_requested || !er.issued) continue;
                auto es = deps_.elm->state(er.key, er.target_gpu);
                const bool dropped =
                    es.gpu_tier == GpuTier::kAbsent && es.interest_count == 0;
                const bool reachable = es.host_tier == HostTier::kWarm ||
                                       es.host_tier == HostTier::kLoadingToRam;
                if (dropped && reachable) {
                    // TD-ORCH-ELM-COMPLETION-LIVELOCK: re-drive ONLY when the
                    // reserve can actually succeed. Every failed
                    // ensure_resident emits a failed lifecycle completion, so
                    // the old unconditional per-cycle retry against a full
                    // stable zone flooded the completion ring with millions
                    // of failures for ONE entry (and held the command open
                    // for the full deadline). With no free slot, DEMOTE the
                    // entry back to the wave scheduler (issued=false): it is
                    // re-issued budget-checked when capacity appears, or the
                    // command finalizes degraded when it never will — same
                    // graceful degradation as the timeout path.
                    // Transient (zone=1) entries check the STREAMING zone's
                    // free slots; with no streaming room they fall back to
                    // zone=0 first (today's stable path) and are then held to
                    // the same stable feasibility check — a full streaming
                    // zone must demote to the wave scheduler exactly like a
                    // full stable zone, never per-cycle re-drive (the
                    // TD-ORCH-ELM-COMPLETION-LIVELOCK discipline).
                    if (er.zone != 0 && deps_.expert_cache
                        && deps_.cuda_kernels_enabled
                        && deps_.expert_cache->free_slots(
                               er.target_gpu,
                               memory::CacheZone::kStreaming) <= 0) {
                        er.zone = 0;  // overflow → stable fallback
                    }
                    const bool can_reserve =
                        er.zone != 0 || !deps_.expert_cache
                        || !deps_.cuda_kernels_enabled
                        || deps_.expert_cache->free_slots(
                               er.target_gpu, memory::CacheZone::kStable) > 0;
                    if (can_reserve) {
                        deps_.elm->ensure_resident(
                            er.key, er.target_gpu, to_zone(er.zone),
                            st.cmd_seq, /*priority=*/kDemandFetchPriority,
                            /*delay_us=*/0);
                    } else {
                        er.issued = false;
                        spdlog::warn(
                            "progressive MoE: fetch L{}E{} gpu={} dropped "
                            "with stable zone full (0 free slots) — demoted "
                            "to wave scheduler instead of per-cycle re-drive "
                            "(cmd_seq {}, layer {})",
                            er.key.layer_idx, er.key.expert_idx,
                            er.target_gpu, st.cmd_seq, st.layer_idx);
                    }
                }
            }
        }

        // TD-FAR-MULTICOMMIT: no incremental compute on new arrivals. Newly
        // arrived experts are tracked (arrived_count) and locked above so the
        // single kFull finalize pass picks them up from live residency; running
        // an incremental pass here would only redo work and risk committing the
        // MoE residual more than once per layer.
        (void)any_new;

        // ── TD-PREFILL-FETCH-SEAM-SCALING: rolling-wave boundary ────────
        // When every ISSUED fetch has arrived but un-issued requested experts
        // remain (the union exceeded the free stable capacity), run a wave-
        // partial compute pass over the arrived-and-uncomputed experts, then
        // (in kWaveDrain) evict them and issue the next wave. Falls through to
        // the legacy termination logic when waves don't apply.
        {
            bool issued_pending = false, unissued_remaining = false;
            for (const auto& er : st.experts) {
                if (er.is_arrived || !er.fetch_requested) continue;
                if (er.issued) issued_pending = true;
                else unissued_remaining = true;
            }
            if (!issued_pending && unissued_remaining
                && st.arrived_count + st.skipped_count < st.total_experts) {
                if (run_moe_wave_pass(st)) {
                    st.phase = ProgressiveMoePhase::kWaveDrain;
                    return false;
                }
                // Nothing computable this wave (no uncomputed arrivals) — try
                // to issue directly into whatever slots are free now. If not a
                // single fetch can be issued the command cannot make progress:
                // finalize with what arrived (graceful degradation, same as
                // the timeout path) instead of burning the deadline.
                if (issue_moe_wave(st) == 0) {
                    spdlog::warn("progressive MoE: no capacity to issue any of "
                                 "the remaining routed experts (layer {}) — "
                                 "finalizing degraded", st.layer_idx);
                    st.phase = ProgressiveMoePhase::kFinalize;
                }
                if (st.phase != ProgressiveMoePhase::kFinalize) return false;
            }
        }

        // Check termination conditions. Skipped experts (F-6 decider) never
        // arrive, so count them toward completion alongside resident experts.
        if (st.phase == ProgressiveMoePhase::kFinalize) {
            // wave logic above already decided to finalize
        } else if (st.arrived_count + st.skipped_count >= st.total_experts) {
            st.phase = ProgressiveMoePhase::kFinalize;
        } else if (!any_fetch_in_flight(st)) {
            // TD-far: finalize the instant nothing requested is still genuinely
            // in flight. A FETCH_AND_RUN command commonly lists the FULL expert
            // set (all N routed experts) even though only the top-K are actually
            // routed; the unrouted experts are requested but can never become
            // resident (no host miss to chase, parked under slot back-pressure,
            // or evicted by demand fetches). Waiting for arrived+skipped to reach
            // total_experts therefore stalls every layer on the 50 ms deadline
            // (58 layers x ~48 ms = the measured ~2.78 s/token). The compute path
            // (dispatch_moe_all_ranks) already runs over LIVE cache residency, so
            // finalizing now computes the correct routed subset. The deadline
            // below remains as a hard safety bound for the (rare) case where an
            // expert is momentarily mid-transfer across the poll.
            st.phase = ProgressiveMoePhase::kFinalize;
        } else if (st.deadline_ns > 0) {
            auto now_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count());
            if (now_ns >= st.deadline_ns) {
                st.timed_out = true;
                st.phase = ProgressiveMoePhase::kFinalize;
            }
        }
    }

    // ── kFinalize: run final pass with allreduce + residual ────────────

    if (st.phase == ProgressiveMoePhase::kFinalize) {
        perf_trace::record(perf_trace::kMoeFinalizeEnter,
                           static_cast<uint16_t>(st.gpu_idx), st.cmd_seq,
                           st.layer_idx, 0);
        // perf_trace x-ray: time the finalize GPU compute on the kExpertFfn stream.
        // Timing-event pair recorded around the dispatch (start before, end after);
        // elapsed reaped at completion → kComputeGpu. Rank-0 for DCP (barrier-synced
        // ≈ per-GPU compute wall). Off unless tracing + CUDA + stream_manager.
        void* compute_t_start = nullptr;
        void* compute_t_end   = nullptr;
        int compute_evt_gpu = static_cast<int>(st.gpu_idx);
        if (deps_.dcp_communicator && deps_.dcp_communicator->is_active()
            && deps_.dcp_executor && !deps_.dcp_executor->gpus().empty())
            compute_evt_gpu = deps_.dcp_executor->gpus()[0].position;
        const bool time_compute = (perf_trace::enabled() || moe_big_xray_enabled())
            && deps_.cuda_kernels_enabled && deps_.stream_manager
            && compute_evt_gpu >= 0
            && compute_evt_gpu < static_cast<int>(deps_.device_backends.size());
        // Final pass: kFull includes allreduce + residual add.
        if (deps_.cuda_kernels_enabled) {
            const int n_experts = deps_.live_config
                ? deps_.live_config->model.n_routed_experts : 0;

            if (n_experts > 0) {
                // Zero the bitset of every GPU this command may dispatch on:
                // the primary + every TP rank (dispatch_moe_all_ranks now
                // HONORS precomputed bitsets — TD-PREFILL-FETCH-SEAM-SCALING —
                // so all ranks' bitsets must be valid, not just the primary's).
                auto zero_bitset = [&](int g) {
                    if (g < 0 || static_cast<size_t>(g) >= moe_scratch_.size())
                        return;
                    auto& b = moe_scratch_[g].expert_resident_bitset;
                    std::memset(b.data(), 0, b.size());
                };
                zero_bitset(static_cast<int>(st.gpu_idx));
                if (deps_.dcp_executor)
                    for (const auto& g : deps_.dcp_executor->gpus())
                        zero_bitset(g.position);
                // INV-MOE-EP-XTP: expert-only ranks' bitsets are populated by
                // the er.target_gpu loop below — zero them too so stale bits
                // from a prior layer never leak into their dispatch.
                for (int g : ep_xtp_gpus_)
                    zero_bitset(g);
                for (const auto& er : st.experts) {
                    // Rolling waves: experts already accumulated by a wave-
                    // partial pass are EXCLUDED (their rows live in
                    // moe_wave_accum; recomputing them would double-count).
                    if (er.is_arrived && !er.computed
                        && er.target_gpu >= 0
                        && static_cast<size_t>(er.target_gpu) < moe_scratch_.size()) {
                        auto& b = moe_scratch_[er.target_gpu].expert_resident_bitset;
                        int eidx = er.key.expert_idx;
                        b[eidx / 8] |= static_cast<uint8_t>(1u << (eidx % 8));
                    }
                }

                InternalMoeParams mp{};
                mp.layer_idx = st.layer_idx;
                mp.num_seqs  = st.num_seqs;
                mp.gpu_idx   = st.gpu_idx;
                mp.moe_mode  = st.moe_mode;
                mp.phase     = MoeDispatchPhase::kFull;
                mp.bitset_precomputed = true;
                // TD-FAR-GATING: consume the precomputed top-K that attention's
                // emit_gating wrote into moe_scratch_ (the same buffer the routed
                // expert list was derived from), instead of re-running the router
                // inside dispatch_moe_internal. The validated RUN_MOE path sets
                // this (dispatch_fused_moe, from run_moe.use_precomputed_gating);
                // without it the FETCH compute self-gates and can compute against
                // a routing that differs from the fetched expert set — numerically
                // diverging from RUN_MOE. The harness exports gating via attention
                // (emit_gating/store_gating) on every FETCH MoE layer, so the
                // top-K buffer is populated and stream-ordered before this pass.
                mp.use_precomputed_gating = true;
                // §12h: the FETCH finalize reports misses from
                // ProgressiveMoeState — the TD-89m D2H probe is redundant here
                // and its stream spin-sync is the largest single host-dispatch
                // item (LS_MOE_SKIP_MISS_PROBE=0 restores it for A/B).
                static const bool skip_probe = [] {
                    const char* v = std::getenv("LS_MOE_SKIP_MISS_PROBE");
                    return !(v && v[0] == '0');
                }();
                mp.skip_miss_probe = skip_probe;
                // TD-PREFILL-FETCH-SEAM-SCALING: when earlier waves accumulated
                // rows, the final pass must add its own rows and unpermute from
                // the accumulator (kFinal). Single-wave commands (union fits
                // capacity — decode, goldens) keep kNone = the legacy path.
                mp.wave_pass = st.wave_ran ? MoeWavePass::kFinal
                                           : MoeWavePass::kNone;
                mp.chunk_tokens = st.chunk_tokens;  // TD-PREFILL-MOE-BIG

                if (time_compute) {
                    auto* be = deps_.device_backends[compute_evt_gpu];
                    compute_t_start = be->create_timing_event();
                    if (compute_t_start)
                        be->record_event(compute_t_start, deps_.stream_manager->stream(
                            compute_evt_gpu, compute::StreamId::kExpertFfn));
                }

                if (deps_.dcp_communicator
                    && deps_.dcp_communicator->is_active()) {
                    dispatch_moe_all_ranks(mp);
                } else {
                    dispatch_moe_internal(mp);
                }

                if (time_compute && compute_t_start) {
                    auto* be = deps_.device_backends[compute_evt_gpu];
                    compute_t_end = be->create_timing_event();
                    if (compute_t_end)
                        be->record_event(compute_t_end, deps_.stream_manager->stream(
                            compute_evt_gpu, compute::StreamId::kExpertFfn));
                }
            }
        }
        perf_trace::record(perf_trace::kMoeFinalizeExit,
                           static_cast<uint16_t>(st.gpu_idx), st.cmd_seq,
                           st.layer_idx, 0);

        // TD-V4-SPEC-PREFILL-CTX: chunk-final aux capture — when this was
        // the LAST layer and the dspark epoch awaits only the final slot,
        // fire the head-sited tap from the committed final residual (the
        // pair attn_buf) so headless prefill chunks arm the draft context.
        // No-op unless the V4 dflash stream-mean arm is live.
        if (deps_.dspark)
            maybe_dspark_capture_moe_final(st.gpu_idx, st.layer_idx,
                                           static_cast<int>(st.num_seqs));

        // Report miss count.
        uint8_t miss_count = static_cast<uint8_t>(
            std::min(st.total_experts - st.arrived_count, 255));

        // TD-FAR-ASYNC: the kFull dispatch above only ENQUEUES the MoE kernels
        // (and the hidden-state commit) on the per-rank kExpertFfn streams — they
        // are still running asynchronously on the GPU. RUN_MOE (dispatch_attention
        // .cpp) never signals CMP_COMPUTE_DONE inline: it records a kExpertFfn
        // event and pushes a PendingCompute so the daemon's poll defers the
        // completion until the GPU work actually finishes. The progressive path
        // signalled completion INLINE here, so the harness saw CMP_COMPUTE_DONE
        // and ran the next stage (and, at end of token, OUTPUT_HEAD which reads
        // hidden_state.attn host-visibly via the output-head stream) while this
        // layer's MoE residual was still in flight — a read-before-write race that
        // produced over-sharp logits (prob 0.958 vs RUN_MOE's 0.493). The race was
        // masked whenever a synchronous D2H dump sat between commands. Defer the
        // completion behind a kExpertFfn event exactly like RUN_MOE so the routed
        // MoE is GPU-complete before the orchestrator/harness proceeds.
        if (deps_.cuda_kernels_enabled && deps_.stream_manager) {
            int event_gpu = static_cast<int>(st.gpu_idx);
            if (deps_.dcp_communicator && deps_.dcp_communicator->is_active()
                && deps_.dcp_executor && !deps_.dcp_executor->gpus().empty()) {
                event_gpu = deps_.dcp_executor->gpus()[0].position;
            }
            void* event = create_and_record_event(
                event_gpu, compute::StreamId::kExpertFfn);

            PendingCompute pc{};
            pc.cmd_seq           = st.cmd_seq;
            pc.gpu_idx           = static_cast<uint32_t>(event_gpu);
            // E_CMD_FAR_FORWARD_LAYER delegation: echo the fused command's
            // type + carry the deduped entry count in data_bytes.
            pc.cmd_type          = st.cmp_cmd_type_override
                                       ? st.cmp_cmd_type_override
                                       : (st.big ? ipc::E_CMD_FETCH_AND_RUN_MOE_BIG
                                                 : ipc::E_CMD_FETCH_AND_RUN_MOE);
            pc.data_bytes        = st.cmp_data_bytes;
            pc.layer_idx         = st.layer_idx;
            pc.cuda_event        = event;
            pc.routed_miss_count = miss_count;
            pc.compute_t_start   = compute_t_start;  // perf_trace: → kComputeGpu on reap
            pc.compute_t_end     = compute_t_end;
            // Union-aware cache partitioning: release the transient streaming
            // residents when this completion is reaped (GPU done reading).
            pc.transient_sweep_mask = st.transient_gpu_mask;
            // LS_MOE_BIG_XRAY: hand the per-command decomposition to the reap
            // (which adds the finalize GPU elapsed and logs one line).
            if (moe_big_xray_enabled() && st.big) {
                PendingCompute::MoeBigXray x;
                x.num_seqs    = st.num_seqs;
                x.experts     = st.total_experts;
                x.waves       = st.xray_waves;
                x.fetches     = st.xray_fetches;
                x.issue_ms    = static_cast<float>(st.xray_issue_ns) / 1e6f;
                x.enq_ms      = static_cast<float>(st.xray_wave_enqueue_ns) / 1e6f;
                x.wait_ms     = static_cast<float>(st.xray_wait_experts_ns) / 1e6f;
                x.drain_ms    = static_cast<float>(st.xray_wave_drain_ns) / 1e6f;
                x.wave_gpu_ms = st.xray_wave_gpu_ms;
                x.wall_ms     = st.xray_enter_ns
                    ? static_cast<float>(xray_now_ns() - st.xray_enter_ns) / 1e6f
                    : 0.0f;
                pc.moe_big_xray = x;
            }
            pending_compute_.push_back(pc);
        } else {
            // Null/test backend: no GPU work to await — signal inline.
            write_compute_completion(
                st.cmp_cmd_type_override
                    ? st.cmp_cmd_type_override
                    : (st.big ? ipc::E_CMD_FETCH_AND_RUN_MOE_BIG
                              : ipc::E_CMD_FETCH_AND_RUN_MOE), st.cmd_seq,
                st.gpu_idx, st.layer_idx, /*status=*/0,
                /*host_buf_offset=*/0, /*data_bytes=*/st.cmp_data_bytes,
                /*top1_prob=*/0.0f, /*entropy=*/0.0f,
                miss_count);
            // No deferred event: sweep the transient streaming residents now.
            if (st.transient_gpu_mask != 0)
                sweep_transient_streaming(st.transient_gpu_mask);
        }

        // Unlock all locked experts. Safe to do now even though the completion is
        // deferred: the harness/orchestrator blocks on CMP_COMPUTE_DONE before it
        // can issue the next FETCH command (whose eviction step is the only thing
        // that could reclaim these slots), and that completion now fires only after
        // the kExpertFfn event is ready — i.e. after the GPU finished reading the
        // expert weights for this layer.
        st.release_locks(deps_.expert_cache);

        // LS_MOE_BIG_XRAY: destroy any un-reaped wave timing pairs (a command
        // finalized without a trailing drain — degraded/error paths).
        for (auto& [gpu, t_start, t_end] : st.wave_timing_events) {
            if (static_cast<size_t>(gpu) < deps_.device_backends.size()
                && deps_.device_backends[gpu]) {
                deps_.device_backends[gpu]->destroy_event(t_start);
                deps_.device_backends[gpu]->destroy_event(t_end);
            }
        }
        st.wave_timing_events.clear();

        progressive_moe_.reset();
        return true;
    }

    return false;
}

// ── Union-aware cache partitioning: transient streaming-zone release ────────
// Evict every evictable streaming-zone resident on the GPUs in gpu_mask.
// Metadata-only (slot free-list return, no D2H): transient union weights are
// read-once by the finalize GEMMs, and the caller runs at completion reap —
// after the kExpertFfn event — so the GPU is done with them. request_evict
// (ELM) / ExpertCache::evict refuse locked, interested, or mid-transfer
// entries (e.g. a just-issued streaming prefetch for the NEXT layer), which
// simply survive to the next sweep. The stable zone and its evict board are
// never touched (the residency listener fires on stable membership only).
void CommandDispatcher::sweep_transient_streaming(uint32_t gpu_mask) {
    if (!deps_.expert_cache || gpu_mask == 0) return;
    const int n_gpus = deps_.expert_cache->gpu_count();
    int swept = 0;
    for (int g = 0; g < n_gpus && g < 32; ++g) {
        if (!(gpu_mask & (1u << g))) continue;
        for (const auto& ri : deps_.expert_cache->residency_snapshot(g)) {
            if (ri.zone != memory::CacheZone::kStreaming) continue;
            const bool ok = deps_.elm
                ? deps_.elm->request_evict(ri.key, g)
                : deps_.expert_cache->evict(ri.key, g);
            if (ok) ++swept;
        }
    }
    if (swept > 0)
        spdlog::debug("sweep_transient_streaming: released {} streaming "
                      "resident(s) (mask {:#x})", swept, gpu_mask);
}

}  // namespace layerstorm::daemon
