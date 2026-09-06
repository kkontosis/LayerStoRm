// E_CMD_REEF_ROUTE + E_CMD_FAR_FORWARD_LAYER handlers (2026-08-18).
//
// The daemon-hosted REEF decision service: the SAME extracted decision stack
// (core/gpu_loader/reef_orch.h) the KEEPER52_REEF_ORCH test arm drives, made
// callable over IPC so the Python orchestrator gets the calibrated
// LoaderSolver placement + the 13c-2.0 victim map without porting the solver
// (the P-18 REEF-arm line). See ipc_protocol.h for both command contracts.
//
// CPU-only TU (INV-GPU-1): device work is reached exclusively through the
// deps_ interfaces (StreamManager events, dispatch internals) — no CUDA
// headers, no raw CUDA calls.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compute/graphs/decode_span_graph.h"  // P-29 step 13 spec-verify bypass
#include "model/model_config.h"                // P-29 step 13 linear-layer census
#include "parallelism/dcp_executor.h"          // P-29 step 13 span_graphs()

#include "core/gpu_loader/reef_orch.h"
#include "core/memory/expert_cache.h"
#include "core/memory/pinned_expert_arena.h"
#include "core/perf_trace.h"
#include "compute/stream_manager.h"
#include "config/config_parser.h"
#include "daemon/command_dispatcher.h"
#include "daemon/far_union.h"

#include <spdlog/spdlog.h>

namespace layerstorm::daemon {

namespace gl = layerstorm::gpu_loader;

// The library's sideband views must stay layout-equal to the IPC entries
// (reef_orch.h is core-side and cannot include daemon/ipc_protocol.h).
static_assert(sizeof(gl::ReefEntry) == sizeof(ipc::ExpertPrefetchEntry));
static_assert(offsetof(gl::ReefEntry, layer_idx) ==
              offsetof(ipc::ExpertPrefetchEntry, layer_idx));
static_assert(offsetof(gl::ReefEntry, expert_idx) ==
              offsetof(ipc::ExpertPrefetchEntry, expert_idx));
static_assert(offsetof(gl::ReefEntry, zone) ==
              offsetof(ipc::ExpertPrefetchEntry, zone));
static_assert(offsetof(gl::ReefEntry, gpu_idx) ==
              offsetof(ipc::ExpertPrefetchEntry, gpu_idx));
static_assert(sizeof(gl::ReefVictim) == sizeof(ipc::ExpertEvictionEntry));
static_assert(offsetof(gl::ReefVictim, expert_idx) ==
              offsetof(ipc::ExpertEvictionEntry, expert_idx));
static_assert(offsetof(gl::ReefVictim, gpu_idx) ==
              offsetof(ipc::ExpertEvictionEntry, gpu_idx));

gl::ReefOrch* CommandDispatcher::ensure_reef_service(std::string& errmsg) {
    if (reef_service_) {
        // TD-KVXP-CAPACITY-REPUBLISH: the per-GPU caps are total_slots
        // snapshots, and 44z elastic grants/reclaims change that total at
        // RUNTIME. Every REEF command funnels through here, so a one-u64
        // generation compare per command keeps the placement model's
        // capacity live: refresh the caps (and re-arm the XTP route caps
        // from them) whenever the cache's elastic topology generation
        // moved. Ordering: begin_drain bumps the generation AND removes the
        // draining zone from total_slots at ratchet step (1), while slabs
        // return to KV only at step (5) — and the daemon is single-threaded
        // (the rebalancer ticks between commands) — so the model shrinks
        // strictly before capacity is gone and can never change mid-solve.
        // With the rebalancer OFF the generation never changes and this
        // branch is byte-identical to the boot-frozen behavior.
        if (deps_.expert_cache) {
            const uint64_t gen = deps_.expert_cache->elastic_generation();
            if (gen != reef_caps_generation_) {
                const int tp = static_cast<int>(reef_service_->cap.size());
                std::vector<int> caps(static_cast<size_t>(tp));
                for (int g = 0; g < tp; ++g)
                    caps[static_cast<size_t>(g)] =
                        deps_.expert_cache->total_slots(
                            g, memory::CacheZone::kStable);
                gl::reef_orch_refresh_caps(*reef_service_, caps,
                                           ep_xtp_gpus_);
                reef_caps_generation_ = gen;
                std::string capdesc;
                for (int g = 0; g < tp; ++g)
                    capdesc += (capdesc.empty() ? "" : " ")
                               + std::to_string(g) + ":"
                               + std::to_string(caps[static_cast<size_t>(g)]);
                spdlog::info("[reef-service] capacity republished (elastic "
                             "generation {}): stable slots {} "
                             "(TD-KVXP-CAPACITY-REPUBLISH)", gen, capdesc);
            }
        }
        return reef_service_.get();
    }
    if (reef_service_failed_) {
        errmsg = reef_service_error_;
        return nullptr;
    }
    auto fail = [&](const std::string& why) -> gl::ReefOrch* {
        reef_service_failed_ = true;
        reef_service_error_ = why;
        errmsg = why;
        return nullptr;
    };
    if (!deps_.live_config)
        return fail("reef service: live config unavailable");
    if (!deps_.expert_cache)
        return fail("reef service: expert cache not configured");

    // Calibration path resolution — EXACTLY the engine loader's rule
    // (engine.cpp init: empty → '<weights dir>/gpu_loader_calibration.json';
    // bare/relative → under the weights dir; absolute → verbatim).
    namespace fs = std::filesystem;
    fs::path weights_dir(deps_.live_config->model.weights_path);
    if (!weights_dir.empty() && !fs::is_directory(weights_dir))
        weights_dir = weights_dir.parent_path();
    fs::path cal_path(deps_.live_config->gpu_loader.calibration_path);
    if (cal_path.empty())
        cal_path = weights_dir / "gpu_loader_calibration.json";
    else if (cal_path.is_relative())
        cal_path = weights_dir / cal_path;

    const int tp =
        static_cast<int>(deps_.live_config->hardware.gpus.size());
    std::vector<int> caps(static_cast<size_t>(tp));
    // Latch the elastic topology generation the caps are derived from —
    // the refresh branch above re-reads them when it moves
    // (TD-KVXP-CAPACITY-REPUBLISH).
    reef_caps_generation_ = deps_.expert_cache->elastic_generation();
    for (int g = 0; g < tp; ++g)
        caps[static_cast<size_t>(g)] = deps_.expert_cache->total_slots(
            g, memory::CacheZone::kStable);

    std::string psrc;
    try {
        reef_service_ = gl::make_reef_orch(cal_path.string(), tp,
                                           std::move(caps), &psrc);
    } catch (const std::exception& e) {
        return fail(std::string("reef service: ") + e.what());
    }
    // TD-MOE-PLACEMENT-CAPACITY-CAP: per-layer route caps on the EP-XTP
    // (expert-only, non-DCP) positions. XTP ranks are excluded from wave
    // passes (TD-MOE-EP-XTP-WAVES): their arrived experts hold stable slots
    // until finalize, so one layer's share must fit the stable zone
    // SIMULTANEOUSLY — an over-capacity share cannot stream through a cold
    // cache and finalizes DEGRADED (incomplete expert set). Cap each XTP
    // position at its MEASURED stable-zone slot count (the same
    // total_slots(kStable) already in caps — never a tuned constant); DCP
    // ranks stay uncapped (INV-FAR-WAVE rolling waves stream their
    // over-capacity shares). ep_xtp_gpus_ is populated in the constructor,
    // strictly before this lazy init. LS_REEF_ROUTE_CAP=0 disables
    // (diagnostic off-switch; the degraded finalize + serving retry then
    // remain the only net).
    if (!ep_xtp_gpus_.empty()) {
        const char* rc = std::getenv("LS_REEF_ROUTE_CAP");
        if (rc && *rc == '0') {
            spdlog::warn("[reef-service] XTP route caps DISABLED "
                         "(LS_REEF_ROUTE_CAP=0 diagnostic) — over-capacity "
                         "XTP shares finalize degraded "
                         "(TD-MOE-PLACEMENT-CAPACITY-CAP)");
        } else {
            reef_service_->route_cap.assign(static_cast<size_t>(tp), -1);
            std::string capdesc;
            for (int g : ep_xtp_gpus_) {
                if (g < 0 || g >= tp) continue;
                reef_service_->route_cap[static_cast<size_t>(g)] =
                    reef_service_->cap[static_cast<size_t>(g)];
                capdesc += (capdesc.empty() ? "" : " ") + std::to_string(g)
                           + ":" + std::to_string(
                                 reef_service_->cap[static_cast<size_t>(g)]);
            }
            spdlog::info("[reef-service] XTP route caps armed (pos:slots {}) "
                         "— per-layer shares on wave-excluded ranks capped "
                         "at measured stable-zone capacity "
                         "(TD-MOE-PLACEMENT-CAPACITY-CAP)", capdesc);
        }
    }
    // Bank seam (INV-REEF-BANK, TD-BRIDGE-CPP-GAP Q1 flip, 2026-08-23):
    // the ONE shared epoch-latched PAIRED bank input builder
    // (gl::install_arena_bank_seam) — bank inputs frozen per placement
    // epoch (arena publishes at service construction = first epoch, and
    // republishes only at online-migrator commit boundaries), HBM→CPU
    // affinity pairing always. Replaces the retired live-raw read (per-
    // solve location_node churned the solver's H2D-cost inputs under the
    // migrator → placement/residency churn → the measured 9.85-vs-10.5
    // bridge gap) and the LS_REEF_BANK_{CSV,PAIRED,SNAPSHOT} diagnostic
    // arms this flip supersedes (ledger: bridge_gap_ledger.md).
    // LS_REEF_BANK_LIVE=1 restores the legacy live-raw read (diagnostic
    // off-switch only).
    if (auto* arena = deps_.pinned_arena) {
        const char* live = std::getenv("LS_REEF_BANK_LIVE");
        if (live && *live == '1') {
            reef_service_->bank_node_fn =
                [arena](uint32_t layer, uint16_t expert) -> int {
                    return arena->location_node(
                        memory::ExpertKey{layer, expert});
                };
            spdlog::warn("[reef-service] bank seam = LIVE raw location_node "
                         "(LS_REEF_BANK_LIVE diagnostic — placement quality "
                         "degrades under the online migrator)");
        } else {
            // First epoch: publish here (daemon thread), BEFORE the first
            // solve — the service path never sees the boot-CSV race.
            if (arena->bank_epoch() == 0) arena->publish_bank_snapshot();
            auto* nm = deps_.numa_manager;
            gl::install_arena_bank_seam(
                *reef_service_, arena, [nm](int n) {
                    if (n < 0 || !nm || !nm->node_is_hbm(n)) return n;
                    for (const auto& h : nm->hbm_nodes())
                        if (h.node == n) return h.cpu_affinity_node;
                    return n;
                });
            spdlog::info("[reef-service] bank seam = shared epoch-latched "
                         "paired snapshot (INV-REEF-BANK, epoch {})",
                         arena->bank_epoch());
        }
    }
    // LS_REEF_RELOC_TRACE=1 (TD-BRIDGE-CPP-GAP Q1 sim probe): interleave
    // arena location-change events into the decision-stream dump — "M
    // <solve_count> <kind> <layer> <expert> <old_node> <new_node>" ('C' =
    // migrator commit flip, 'R'/'E' = reserve/evict) — plus one "H <tp>
    // <cap...>" header so the offline replay (tools/reef_sim) can rebuild
    // the exact solver construction. Same daemon thread as the solves, so
    // the interleaving order is the true order. Requires the dump armed.
    if (const char* rt = std::getenv("LS_REEF_RELOC_TRACE");
        rt && *rt == '1' && reef_service_->decision_dump
        && deps_.pinned_arena) {
        std::FILE* df = reef_service_->decision_dump;
        std::fprintf(df, "H %d", tp);
        for (int g = 0; g < tp; ++g)
            std::fprintf(df, " %d", reef_service_->cap[static_cast<size_t>(g)]);
        std::fputc('\n', df);
        auto* svc_ptr = reef_service_.get();
        deps_.pinned_arena->location_change_sink =
            [svc_ptr, df](memory::ExpertKey k, int oldn, int newn, char kind) {
                std::fprintf(df, "M %llu %c %u %u %d %d\n",
                             static_cast<unsigned long long>(
                                 svc_ptr->solve_count),
                             kind, k.layer_idx,
                             static_cast<unsigned>(k.expert_idx), oldn, newn);
            };
        reef_reloc_trace_installed_ = true;
        spdlog::info("[reef-service] arena relocation trace armed "
                     "(LS_REEF_RELOC_TRACE)");
    }
    spdlog::info("[reef-service] armed: calib={} devices={} policy={} "
                 "freq_w={} reuse_w={} reuse_tau={} keeppred_w={}",
                 cal_path.string(), tp, psrc, reef_service_->freq_w,
                 reef_service_->reuse_w, reef_service_->reuse_tau,
                 reef_service_->keeppred_w);
    return reef_service_.get();
}

// Shared REEF placement core: route `topk` for `layer`, rewrite the sideband
// prefetch entries' gpu_idx in place (zone 0), fill the index-aligned victim
// map (sentinel first, then reef_orch_apply). Returns false only on a
// service failure (errmsg set).
static bool reef_place(gl::ReefOrch& svc, uint32_t layer,
                       const std::vector<uint16_t>& topk,
                       ipc::ExpertPrefetchEntry* entries,
                       ipc::ExpertEvictionEntry* evicts,
                       uint32_t trace_seq = 0) {
    const uint32_t n = static_cast<uint32_t>(topk.size());
    std::vector<uint8_t> assign(n);
    gl::reef_orch_route(svc, static_cast<int>(layer), topk, assign);
    perf_trace::record(perf_trace::kFarRouteDone, 0, trace_seq, layer, n);
    for (uint32_t i = 0; i < n; ++i) {
        entries[i].layer_idx  = layer;
        entries[i].expert_idx = topk[i];
        entries[i].zone       = 0;
        entries[i].gpu_idx    = assign[i];
        evicts[i].layer_idx   = layer;
        evicts[i].expert_idx  = 0xFFFF;   // sentinel: no victim
        evicts[i].gpu_idx     = assign[i];
        evicts[i]._pad        = 0;
    }
    gl::reef_orch_apply(svc, static_cast<int>(layer),
                        reinterpret_cast<const gl::ReefEntry*>(entries),
                        reinterpret_cast<gl::ReefVictim*>(evicts), n);
    return true;
}

void CommandDispatcher::handle_reef_route(const ipc::Command& cmd) {
    const auto& p = cmd.reef_route;
    if (!deps_.sideband_base) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "reef_route: sideband not configured");
        return;
    }
    if (p.expert_count == 0 || p.expert_count > ipc::kMaxExpertPrefetch) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "reef_route: expert_count out of range");
        return;
    }
    std::string err;
    auto* svc = ensure_reef_service(err);
    if (!svc) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe, err.c_str());
        return;
    }
    auto* entries = reinterpret_cast<ipc::ExpertPrefetchEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertPrefetchOff);
    auto* evicts = reinterpret_cast<ipc::ExpertEvictionEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertEvictionOff);
    std::vector<uint16_t> topk(p.expert_count);
    for (uint32_t i = 0; i < p.expert_count; ++i)
        topk[i] = entries[i].expert_idx;
    reef_place(*svc, p.layer_idx, topk, entries, evicts);
    // Synchronous CPU-only command — complete inline (the orchestrator's
    // next FETCH_AND_RUN_MOE reads the rewritten sideband in ring order).
    write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                             p.layer_idx, /*status=*/0,
                             /*host_buf_offset=*/0,
                             /*data_bytes=*/p.expert_count);
}

void CommandDispatcher::handle_far_forward_layer(const ipc::Command& cmd) {
    const auto& p = cmd.far_forward_layer;
    if (!deps_.sideband_base || !deps_.live_config) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: sideband/config not configured");
        return;
    }
    if (p.num_seqs == 0 || p.num_seqs > ipc::kMaxBatchDescriptors
        || p.num_seqs > static_cast<uint32_t>(moe_batch_capacity_)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: num_seqs out of range");
        return;
    }
    // Burst-published FAR sweeps (sliding-window orchestrator sends): a
    // still-active progressive MoE from the PREVIOUS layer is backpressure,
    // not an error — drain it to finalization (its completion publishes
    // in order ahead of this command's), then dispatch normally. Identical
    // semantics to serial pacing: layer L+1's attention depends on layer
    // L's MoE output regardless.
    if (!drain_progressive_moe("far_forward_layer")) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: progressive MoE already active");
        return;
    }
    const bool is_moe =
        p.layer_idx >= static_cast<uint32_t>(
            deps_.live_config->model.first_k_dense_replace);

    // ── 1. Attention (+ fused gate & routing export on MoE layers) — the
    // exact D_B_CMD_RUN_ATTENTION internals via a synthesized command.
    //
    // P-29 step 13 phase B (p.spec_verify): the speculative-verify shape runs a
    // PER-ROW loop of B=1 DECODE-shaped dispatches instead of one batched
    // attention: row j reads descriptor j (staged into slot 0 per row),
    // addresses hidden rows at row_offset=j, exports its routed top-K at
    // sideband dst row j (cumulative header), and — on KDA layers — the
    // recurrent state advances through the exact per-token decode kernels
    // (INV-KDA-CARRY: a non-64 chunked-scan cut is only tolerance-equal,
    // so the chunk path is NEVER used here). kda_snap_mask bit j takes a
    // per-layer anchor copy right after row j's state update (pool-
    // boundary crossings; INV-KDA-REWIND anchor-and-replay). Span graphs
    // are bypassed for the whole loop so verify row shapes never spend
    // the kVariantCap variant budget.
    if (p.spec_verify) {
        const uint32_t R = p.num_seqs;
        if (R > 8) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kFetchAndRunMoe,
                        "far_forward_layer: spec_verify rows > 8");
            return;
        }
        auto* be = reinterpret_cast<ipc::BatchDescriptorEntry*>(
            deps_.sideband_base + ipc::IpcLayout::kBatchDescriptorOff);
        ipc::BatchDescriptorEntry saved[8];
        for (uint32_t j = 0; j < R; ++j) saved[j] = be[j];
        for (uint32_t j = 1; j < R; ++j) {
            if (saved[j].seq_id != saved[0].seq_id
                || saved[j].token_pos != saved[0].token_pos + j) {
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kFetchAndRunMoe,
                            "far_forward_layer: spec_verify rows must be "
                            "one sequence at consecutive positions");
                return;
            }
        }
        // Linear-layer bookkeeping for KDA anchors.
        const auto& mcfg = deps_.live_config->model;
        model::ModelConfig mc(mcfg);
        const bool is_linear = mc.is_linear_attention_layer(
            static_cast<int>(p.layer_idx));
        int linear_ord = 0, linear_total = 0;
        if (is_linear || p.kda_snap_mask) {
            for (int l = 0; l < mcfg.num_hidden_layers; ++l) {
                if (!mc.is_linear_attention_layer(l)) continue;
                if (l < static_cast<int>(p.layer_idx)) ++linear_ord;
                ++linear_total;
            }
        }
        SequenceState* st = find_seq(saved[0].seq_id);
        const int kp = std::max(1, mcfg.index_kpool);

        // P-29 step 13: span graphs STAY ON for verify rows — the span keys carry
        // a row axis (make_key rank + 16*row), so each (chain, layer, rank,
        // row) captures its own variants and row-0 (plain decode) keys are
        // never polluted. Replays are safe by the INV-0.6(b) device-read
        // argument, row buffers are per-row-offset stable, and the eager
        // alternative measured ~20 ms/row of host launch time.

        bool ok = true;
        const char* why = nullptr;
        for (uint32_t j = 0; j < R && ok; ++j) {
            be[0] = saved[j];
            ipc::Command ac{};
            ac.cmd_type = ipc::D_B_CMD_RUN_ATTENTION;
            ac.cmd_seq  = cmd.cmd_seq;
            ac.gpu_idx  = cmd.gpu_idx;
            ac.run_attention.layer_idx    = p.layer_idx;
            ac.run_attention.num_seqs     = 1;
            ac.run_attention.is_prefill   = 0;
            ac.run_attention.chunk_start  = 0;
            ac.run_attention.chunk_len    = 0;
            ac.run_attention.emit_gating  = is_moe ? 1 : 0;
            ac.run_attention.store_gating = is_moe ? 1 : 0;
            ac.run_attention.row_offset   = j;
            ac.run_attention.spec_flags   = 1;  // export at dst row j
            if (!dispatch_fused_attention(ac)) { ok = false; break; }
            // KDA anchor snapshot after this row's state update (this
            // layer's span only — other layers copy in their own sweep
            // commands; the anchor commits when the LAST linear layer's
            // copy lands, and invalidates at the first).
            if (is_linear && st && ((p.kda_snap_mask >> j) & 1)) {
                const uint32_t anchor_pos = saved[j].token_pos + 1;
                // Slot 0 = pool-boundary anchors, slot 1 = per-round
                // anchors (see handle_kda_snapshot).
                const int slot = (anchor_pos % kp == 0) ? 0 : 1;
                // Anchor slots are claimed at ADMISSION (seq_create/fork).
                // A missing slot (probe producer / claim-failed fork)
                // gets a best-effort LAZY claim here — a silent skip
                // would poison the orchestrator's anchor mirror and turn
                // a later restore into a request-fatal refusal.
                if (st->kda_anchors.slots[slot].empty()
                    && linear_ord == 0) {
                    std::string aerr;
                    if (!claim_kda_state(saved[0].seq_id,
                                         static_cast<int>(cmd.gpu_idx),
                                         st->kda_anchors.slots[slot],
                                         aerr)) {
                        st->kda_anchors.slots[slot].clear();
                        spdlog::warn(
                            "spec_verify: KDA anchor slot {} claim failed "
                            "for seq {} ({}) — snapshot skipped",
                            slot, saved[0].seq_id, aerr);
                    }
                }
                if (st->kda_anchors.slots[slot].empty()) {
                    // Claim failed: leave pos kNone (the restore will
                    // refuse loudly rather than restore garbage).
                } else {
                    if (linear_ord == 0)
                        st->kda_anchors.pos[slot] =
                            SequenceState::KdaAnchors::kNone;
                    if (!kda_anchor_copy_layer(*st, slot, linear_ord,
                                               /*to_anchor=*/true)) {
                        ok = false;
                        why = "spec_verify: KDA anchor layer copy failed";
                        break;
                    }
                    if (linear_ord == linear_total - 1)
                        st->kda_anchors.pos[slot] = anchor_pos;
                }
            }
        }
        be[0] = saved[0];  // restore the descriptor region
        if (!ok) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        why ? ipc::CmpErrorCategory::kFetchAndRunMoe
                            : (last_internal_error_msg_
                                   ? last_internal_error_cat_
                                   : ipc::CmpErrorCategory::
                                         kComputeValidation),
                        why ? why
                            : (last_internal_error_msg_
                                   ? last_internal_error_msg_
                                   : "far_forward_layer: spec_verify "
                                     "attention dispatch failed"));
            return;
        }
    } else {
    ipc::Command ac{};
    ac.cmd_type = ipc::D_B_CMD_RUN_ATTENTION;
    ac.cmd_seq  = cmd.cmd_seq;
    ac.gpu_idx  = cmd.gpu_idx;
    ac.run_attention.layer_idx    = p.layer_idx;
    ac.run_attention.num_seqs     = p.num_seqs;
    ac.run_attention.is_prefill   = p.is_prefill;
    ac.run_attention.chunk_start  = p.chunk_start;
    ac.run_attention.chunk_len    = p.chunk_len;
    ac.run_attention.emit_gating  = is_moe ? 1 : 0;
    ac.run_attention.store_gating = is_moe ? 1 : 0;
    if (!dispatch_fused_attention(ac)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    last_internal_error_msg_
                        ? last_internal_error_cat_
                        : ipc::CmpErrorCategory::kComputeValidation,
                    last_internal_error_msg_
                        ? last_internal_error_msg_
                        : "far_forward_layer: attention dispatch failed");
        return;
    }
    }
    perf_trace::record(perf_trace::kFarAttnDispatched, 0,
                       static_cast<uint32_t>(cmd.cmd_seq), p.layer_idx, 0);

    // ── 1.5 P-29 step 16 (LS_FAR_PROLOGUE_PREISSUE): pre-issue the routing-
    // independent MoE prologue NOW — the whole attention chain (incl. fused
    // gating top-K + routing export) is already enqueued and attn_moe_event
    // is recorded after all of it, so the per-rank collapse+norm and the
    // EP-XTP broadcast queue safely behind their producers and EXECUTE
    // during the readback spin + union/REEF/decider host window below,
    // instead of serializing after it (the P-29 step-15 measured ~70 us
    // readback->first-MoE-kernel box-empty gap, ~29x/token). Decode-shaped
    // MoE commands only; any failure leaves the flags clear and the
    // finalize path re-emits byte-identically. Non-qualifying commands
    // clear any stale pre-issue so a later dispatch of the same layer at a
    // different shape can never consume a 1-row primed buffer.
    // P-32 stage 1 (LS_SPEC_VERIFY_FETCH_HIDE): spec_verify commands are
    // decode-shaped R<=8 row blocks whose prime pass is shape-parametric
    // (pmp.num_seqs rides through; the C-6 early kick already proves
    // prime_cpu_input_only at M>1). The primed row count is published
    // (far_prologue_num_seqs_) and both consumers match on it, so a primed
    // set can never be consumed at a different shape.
    const bool prologue_shape_ok =
        p.num_seqs == 1
        || (p.spec_verify && p.num_seqs <= 8
            && spec_verify_fetch_hide_enabled());
    if (is_moe && prologue_shape_ok && !p.is_prefill
        && deps_.cuda_kernels_enabled && far_prologue_preissue_enabled()) {
        preissue_far_moe_prologue(p.layer_idx, p.num_seqs, cmd.gpu_idx);
    } else {
        clear_far_prologue();
    }

    // ── 2. Daemon-side wait for the attention stream — covers the
    // routing-export D2H (issued on the same stream), replacing the
    // orchestrator's external attention-completion wait.
    if (deps_.cuda_kernels_enabled && deps_.stream_manager) {
        int event_gpu = static_cast<int>(cmd.gpu_idx);
        if (deps_.dcp_executor && !deps_.dcp_executor->gpus().empty())
            event_gpu = deps_.dcp_executor->gpus()[0].position;
        void* ev = create_and_record_event(event_gpu,
                                           compute::StreamId::kAttention);
        for (;;) {
            auto [status, verr] =
                deps_.stream_manager->query_event(ev, event_gpu);
            if (status == compute::EventStatus::kReady) break;
            if (status == compute::EventStatus::kError) {
                deps_.stream_manager->destroy_event(ev, event_gpu);
                char msg[80];
                std::snprintf(msg, sizeof(msg),
                              "far_forward_layer: attention stream error "
                              "(vendor err=%d)", verr);
                write_error(cmd.cmd_seq, cmd.gpu_idx,
                            ipc::CmpErrorCategory::kGpuFatal, msg);
                return;
            }
        }
        deps_.stream_manager->destroy_event(ev, event_gpu);
    }
    perf_trace::record(perf_trace::kFarAttnReady, 0,
                       static_cast<uint32_t>(cmd.cmd_seq), p.layer_idx, 0);

    // ── Dense layer: the dense-FFN path (D_B_CMD_RUN_MOE internals),
    // single completion on the MoE stream, data_bytes=0.
    if (!is_moe) {
        ipc::Command mc{};
        mc.cmd_type = ipc::D_B_CMD_RUN_MOE;
        mc.cmd_seq  = cmd.cmd_seq;
        mc.gpu_idx  = cmd.gpu_idx;
        mc.run_moe.layer_idx = p.layer_idx;
        mc.run_moe.num_seqs  = p.num_seqs;
        if (!dispatch_fused_moe(mc)) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        last_internal_error_msg_
                            ? last_internal_error_cat_
                            : ipc::CmpErrorCategory::kComputeValidation,
                        last_internal_error_msg_
                            ? last_internal_error_msg_
                            : "far_forward_layer: dense MoE dispatch failed");
            return;
        }
        if (deps_.cuda_kernels_enabled) {
            void* event = create_and_record_event(
                static_cast<int>(cmd.gpu_idx),
                compute::StreamId::kExpertFfn);
            PendingCompute pc{};
            pc.cmd_seq    = cmd.cmd_seq;
            pc.gpu_idx    = cmd.gpu_idx;
            pc.cmd_type   = cmd.cmd_type;
            pc.layer_idx  = p.layer_idx;
            pc.cuda_event = event;
            // TD-INDEXER-NO-DENSE-FALLBACK witness byte (attention ran
            // above under this command).
            pc.indexer_dense = step_indexer_dense_;
            step_indexer_dense_ = 0;
            pending_compute_.push_back(pc);
        } else {
            const uint8_t idense = step_indexer_dense_;
            step_indexer_dense_ = 0;
            write_compute_completion(cmd.cmd_type, cmd.cmd_seq, cmd.gpu_idx,
                                     p.layer_idx, /*status=*/0,
                                     /*host_buf_offset=*/0, /*data_bytes=*/0,
                                     /*top1_prob=*/0.0f, /*entropy=*/0.0f,
                                     /*routed_miss_count=*/0,
                                     /*moe_degraded=*/0,
                                     /*indexer_dense=*/idense);
        }
        return;
    }

    // ── 3. Deduped first-occurrence routed union from the export. ────────
    const auto* hdr = reinterpret_cast<const ipc::RoutingExportHeader*>(
        deps_.sideband_base + ipc::IpcLayout::kRoutingExportOff);
    if (hdr->num_tokens != p.num_seqs
        || hdr->layer_idx != p.layer_idx) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: routing export mismatch");
        return;
    }
    const auto* ridx = reinterpret_cast<const int32_t*>(
        deps_.sideband_base + ipc::IpcLayout::kRoutingExportIndicesOff);
    const uint32_t rn = hdr->num_tokens * hdr->topk;
    const int num_experts = deps_.live_config->model.n_routed_experts;
    std::vector<uint16_t> topk;
    topk.reserve(64);
    // TD-GLM5N-ROUTED-EXPERT-ID-TRUNCATION: the union builder must never
    // silently thin the routed set (kMaxExperts >= every shipped
    // n_routed_experts, enforced at boot); a cap violation here is a loud
    // error, not a skip.
    if (!build_routed_union(ridx, rn, num_experts, topk)) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: n_routed_experts exceeds "
                    "ipc::kMaxExperts (routed union would truncate)");
        return;
    }
    if (topk.empty()) {
        write_error(cmd.cmd_seq, cmd.gpu_idx,
                    ipc::CmpErrorCategory::kFetchAndRunMoe,
                    "far_forward_layer: no routed experts exported");
        return;
    }
    const uint32_t count = static_cast<uint32_t>(topk.size());
    perf_trace::record(perf_trace::kFarUnionDone, 0,
                       static_cast<uint32_t>(cmd.cmd_seq), p.layer_idx, count);

    // ── 4. Placement per route_mode → sideband entries (+ victim map). ──
    auto* entries = reinterpret_cast<ipc::ExpertPrefetchEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertPrefetchOff);
    auto* evicts = reinterpret_cast<ipc::ExpertEvictionEntry*>(
        deps_.sideband_base + ipc::IpcLayout::kExpertEvictionOff);
    bool have_evict_map = false;
    if (p.route_mode == 1) {
        std::string err;
        auto* svc = ensure_reef_service(err);
        if (!svc) {
            write_error(cmd.cmd_seq, cmd.gpu_idx,
                        ipc::CmpErrorCategory::kFetchAndRunMoe, err.c_str());
            return;
        }
        reef_place(*svc, p.layer_idx, topk, entries, evicts,
                   static_cast<uint32_t>(cmd.cmd_seq));
        have_evict_map = true;
    } else {
        // ACT arm: static e % <expert hosts>.  Expert hosts = the
        // expert-role PREFIX of hardware.gpus (a GPU whose roles carry
        // neither resident nor expert_streaming — e.g. a dedicated draft
        // host — stops the scan; the schema default gives a role-less GPU
        // all roles).  Mirrors the bridge's moe_gpus rule exactly so the
        // FAR act arm places byte-identically to the split act arm.
        const auto& gpus = deps_.live_config->hardware.gpus;
        uint32_t tp = 0;
        for (const auto& g : gpus) {
            bool expert_host = false;
            for (const auto r : g.roles)
                if (r == config::GpuRole::resident
                    || r == config::GpuRole::expert_streaming) {
                    expert_host = true;
                    break;
                }
            if (!expert_host) break;
            ++tp;
        }
        for (uint32_t i = 0; i < count; ++i) {
            entries[i].layer_idx  = p.layer_idx;
            entries[i].expert_idx = topk[i];
            entries[i].zone       = 0;
            entries[i].gpu_idx    =
                static_cast<uint8_t>(topk[i] % (tp ? tp : 1));
        }
    }

    perf_trace::record(perf_trace::kFarSolveDone, 0,
                       static_cast<uint32_t>(cmd.cmd_seq), p.layer_idx, count);

    // ── 5. The production FETCH_AND_RUN execution path, delegated under
    // THIS command's seq — its finalize completion echoes our cmd_type and
    // carries the entry count (cmp_* overrides on the progressive state).
    // Synthesized with OUR cmd_type: the impl reads the payload through the
    // union (layout-identical) and sets the completion overrides at
    // state-build — required because the num_seqs==1 gated-final path
    // finalizes inside the call (post-return patching would miss decode
    // layers; the count must ride EVERY routed-MoE FAR completion).
    ipc::Command fc{};
    fc.cmd_type = ipc::E_CMD_FAR_FORWARD_LAYER;
    fc.cmd_seq  = cmd.cmd_seq;
    fc.gpu_idx  = cmd.gpu_idx;
    fc.fetch_and_run_moe.layer_idx      = p.layer_idx;
    fc.fetch_and_run_moe.num_seqs       = p.num_seqs;
    fc.fetch_and_run_moe.expert_count   = count;
    fc.fetch_and_run_moe.timeout_us     = p.timeout_us;
    fc.fetch_and_run_moe.moe_mode       = 0;
    fc.fetch_and_run_moe.have_evict_map = have_evict_map ? 1 : 0;
    handle_fetch_and_run_moe_impl(fc, /*big=*/false,
                                  /*spec_verify=*/p.spec_verify != 0);
}

}  // namespace layerstorm::daemon
