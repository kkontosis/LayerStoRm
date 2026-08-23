// MoE multi-rank orchestration — TP all-ranks dispatch with allreduce,
// EP extra-rank broadcast and extras dispatch (INV-MOE-EP-XTP).
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

// ── KD-4g: Multi-rank MoE dispatch with TP allreduce ────────────────────────

bool CommandDispatcher::dispatch_moe_all_ranks(const InternalMoeParams& mp_template) {
    if (!deps_.dcp_executor || !deps_.dcp_communicator) {
        return dispatch_moe_internal(mp_template);
    }

    const auto& tp_gpus = deps_.dcp_executor->gpus();
    const int dcp_size = deps_.dcp_executor->dcp_size();

    if (dcp_size <= 1) {
        return dispatch_moe_internal(mp_template);
    }

    if (!deps_.live_config) return false;

    const int first_k_dense = deps_.live_config->model.first_k_dense_replace;
    const bool is_dense = (static_cast<int>(mp_template.layer_idx) < first_k_dense);
    const int num_tokens = static_cast<int>(mp_template.num_seqs);

    // C-6 early-kick η: the fetch window CLOSES at finalize entry (experts have
    // arrived ⇒ the missing-expert H2D is done). The early kick's host FFN ran
    // during [win_start, win_end] ⇒ genuine overlap; the late kick's host FFN is
    // spawned below at Phase-1 (after this stamp) ⇒ starts past win_end ⇒ η→0.
    if (mp_template.phase == MoeDispatchPhase::kFull
        && cpu_layer_has_forced(mp_template.layer_idx))
        cpu_fetch_win_end_ns_ = xray_now_ns();

    // 13c-7 / INV-MoE-TP: For TP>1 non-dense layers, build per-GPU resident
    // expert bitsets and compare across ranks.
    //  - Identical bitsets → old path: intersect (identity) so all GPUs run the
    //    same set. No routed allreduce needed (each GPU produces identical output).
    //  - Non-identical bitsets → EP-within-TP: each GPU keeps its own bitset and
    //    runs its own expert subset. Phase 2 allreduces moe_output to combine
    //    partial routed sums. Orchestrator ensures disjoint assignment.
    bool bitsets_precomputed = false;
    bool ep_within_tp = false;
    // INV-MOE-EP-XTP: expert-only (non-DCP) GPUs whose resident bitsets are
    // non-empty this layer — they run their routed subsets and D2D-fold their
    // partials onto a TP rank (dispatch_moe_ep_extras). Empty when EP == TP.
    std::vector<int> active_extras;
    if (!is_dense && deps_.expert_cache) {
        const int n_experts = deps_.live_config->model.n_routed_experts;
        const size_t bitset_bytes = static_cast<size_t>((n_experts + 7) / 8);

        // Build per-GPU bitsets from expert cache residency.
        // TD-PREFILL-FETCH-SEAM-SCALING: when the caller PRE-POPULATED every TP
        // rank's bitset (the progressive FETCH finalize — arrived ∧ not yet
        // wave-computed), honor it instead of rebuilding from live residency:
        // rows of experts already accumulated by earlier waves (and evicted)
        // must NOT be recomputed (double count), and a still-resident computed
        // expert must be excluded for the same reason. The ep_within_tp
        // detection/intersection/dedup below operate on the given bitsets.
        if (!mp_template.bitset_precomputed) {
        auto build_live_bitset = [&](int gpu_pos) {
            if (gpu_pos < 0 || static_cast<size_t>(gpu_pos) >= moe_scratch_.size())
                return;
            auto& bs = moe_scratch_[gpu_pos].expert_resident_bitset;
            std::memset(bs.data(), 0, bs.size());
            for (int e = 0; e < n_experts; ++e) {
                memory::ExpertKey key{static_cast<uint32_t>(mp_template.layer_idx),
                                      static_cast<uint16_t>(e)};
                const auto* entry = deps_.expert_cache->lookup(key, gpu_pos);
                if (entry && (entry->sub_components_ready & memory::SubComponent::kAll)
                              == memory::SubComponent::kAll) {
                    bs[e / 8] |= static_cast<uint8_t>(1u << (e % 8));
                }
            }
        };
        for (int r = 0; r < dcp_size; ++r)
            build_live_bitset(tp_gpus[r].position);
        // INV-MOE-EP-XTP: expert-only GPUs' residency too (self-gating legacy
        // path; the FETCH finalize pre-populates these from er.target_gpu).
        for (int g : ep_xtp_gpus_)
            build_live_bitset(g);
        }

        // INV-MOE-EP-XTP: collect the expert-only GPUs with routed residents
        // this layer. Any non-empty extra ⇒ the cross-rank routed combine MUST
        // run (their partials fold onto a TP rank pre-combine), so this forces
        // EP mode exactly like differing TP bitsets do.
        if (deps_.cuda_kernels_enabled) {
            for (int g : ep_xtp_gpus_) {
                if (g < 0 || static_cast<size_t>(g) >= moe_scratch_.size())
                    continue;
                const auto& bs = moe_scratch_[g].expert_resident_bitset;
                bool any = false;
                for (size_t j = 0; j < bitset_bytes; ++j)
                    if (bs[j]) { any = true; break; }
                // INV-MOE-OVERLAP: an extra rank whose experts were ALL
                // computed by the resident-overlap pass has an empty finalize
                // bitset but rows in its wave accumulator — it must still
                // dispatch (kFinal unpermutes the accumulator) and fold, or
                // its contribution is silently dropped.
                if (!any && mp_template.wave_pass == MoeWavePass::kFinal
                    && static_cast<size_t>(g) < moe_wave_accum_used_.size()
                    && moe_wave_accum_used_[g])
                    any = true;
                if (any) active_extras.push_back(g);
            }
            if (!active_extras.empty()) ep_within_tp = true;
        }

        // Compare bitsets across all TP GPUs to detect EP-within-TP.
        {
            const int ref_pos = tp_gpus[0].position;
            if (ref_pos >= 0 && static_cast<size_t>(ref_pos) < moe_scratch_.size()) {
                const auto& ref_bs = moe_scratch_[ref_pos].expert_resident_bitset;
                for (int r = 1; r < dcp_size; ++r) {
                    const int gpu_pos = tp_gpus[r].position;
                    if (gpu_pos < 0 || static_cast<size_t>(gpu_pos) >= moe_scratch_.size())
                        continue;
                    const auto& bs = moe_scratch_[gpu_pos].expert_resident_bitset;
                    if (std::memcmp(ref_bs.data(), bs.data(), bitset_bytes) != 0) {
                        ep_within_tp = true;
                        break;
                    }
                }
            }
        }

        // TD-PREFILL-FETCH-SEAM-SCALING: a rolling-wave FINAL pass carries
        // rank-DISJOINT accumulated expert rows (each expert was fetched+
        // computed on exactly one rank) even when the final-pass bitsets
        // happen to compare identical (e.g. both empty because every arrival
        // was already wave-computed). The routed cross-rank combine MUST run,
        // and the identical-bitset intersection must not zero the sets.
        if (mp_template.wave_pass == MoeWavePass::kFinal)
            ep_within_tp = true;

        // ── TD-MOE-EP-COMBINE-RESIDUAL: DIAGNOSTIC fork-trace (off by default) ──
        // LS_FORK_TRACE: log, per non-dense routed layer-call, whether the cross-GPU
        // partition is EP-within-TP (bitsets differ ⇒ canonical per-slot combine) or
        // the identical-bitset "replicated" fallback (ep_within_tp=FALSE ⇒ the
        // placement-DEPENDENT mode-0 bf16 partial-sum combine). Also dumps per-GPU
        // resident counts. A call index N maps to decode-step N/(#non-dense layers).
        // Pure visibility; zero behavior effect.
        static const bool fork_trace = [] {
            const char* v = std::getenv("LS_FORK_TRACE");
            return v && v[0] && v[0] != '0';
        }();
        if (fork_trace) {
            static std::atomic<uint64_t> fcall{0};
            const uint64_t fc = fcall.fetch_add(1);
            int rc[8] = {0};
            for (int r = 0; r < dcp_size && r < 8; ++r) {
                const int gp = tp_gpus[r].position;
                if (gp < 0 || static_cast<size_t>(gp) >= moe_scratch_.size()) continue;
                const auto& bs = moe_scratch_[gp].expert_resident_bitset;
                int c = 0;
                for (int e = 0; e < n_experts; ++e) if ((bs[e/8] >> (e%8)) & 1) ++c;
                rc[r] = c;
            }
            spdlog::warn("LS_FORK_TRACE call={} layer={} ep_within_tp={} resident_g0={} resident_g1={}",
                         fc, mp_template.layer_idx, ep_within_tp ? 1 : 0, rc[0], rc[1]);
        }

        if (!ep_within_tp) {
            // Old path: bitsets identical — intersect for safety (handles edge
            // cases where cache state changes between build and dispatch).
            std::vector<uint8_t> intersection(bitset_bytes, 0xFF);
            for (int r = 0; r < dcp_size; ++r) {
                const int gpu_pos = tp_gpus[r].position;
                if (gpu_pos < 0 || static_cast<size_t>(gpu_pos) >= moe_scratch_.size())
                    continue;
                const auto& bs = moe_scratch_[gpu_pos].expert_resident_bitset;
                for (size_t j = 0; j < bitset_bytes; ++j)
                    intersection[j] &= bs[j];
            }
            for (int r = 0; r < dcp_size; ++r) {
                const int gpu_pos = tp_gpus[r].position;
                if (gpu_pos < 0 || static_cast<size_t>(gpu_pos) >= moe_scratch_.size())
                    continue;
                auto& bs = moe_scratch_[gpu_pos].expert_resident_bitset;
                std::copy(intersection.begin(), intersection.end(), bs.begin());
            }
        }
        // EP path: each GPU keeps its own bitset (no intersection).
        bitsets_precomputed = true;

        // ── INV-MOE-EP-DISJOINT enforcement — UNCONDITIONAL dedup (Phase-1c) ──
        // EP-within-TP keeps each GPU's residency bitset independently and runs
        // EVERY GPU over its resident subset; the per-token cross-GPU combine
        // (legacy bf16 sum OR the canonical per-slot SUM-gather) ASSUMES each
        // routed expert is resident on exactly ONE GPU (disjoint EP). Non-disjoint
        // placement (the ACT reroute, dispatch_loader.cpp:278, fetches a
        // home-missing expert to a NON-e%tp GPU while a stale home copy persists)
        // makes the SAME expert resident on >1 GPU (a cross-GPU DUPLICATE); every
        // holder computes it and the SUM combine DOUBLE-COUNTS its contribution
        // ((#holders)·w_k·expert_out_k) — a real correctness bug, not a config
        // knob. We therefore ALWAYS enforce a single canonical owner: the
        // LOWEST-rank holder keeps each duplicated expert; the others skip it, so
        // each c_k is counted exactly once (TD-MOE-EP-COMBINE-RESIDUAL).
        //
        // COLLECTIVE-FREE: ownership is decided purely from the per-rank residency
        // bitsets THIS thread already built above (from each rank's own expert
        // cache) — NO new cross-GPU collective / sync on the MoE hot path. On the
        // disjoint path (production RUN_MOE, baseline e%tp) there are zero
        // duplicates so the dedup is a byte-identical no-op.
        //
        // LS_EP_DUP_DUMP: observe-only telemetry (default off) — counts cross-GPU
        // duplicate experts per (call,layer); pure visibility, no behavior effect.
        if (ep_within_tp && dcp_size >= 2) {
            static constexpr int kMaxRanks = 8;  // mirrors kMaxTp (EP ≤ 8 ranks)
            uint8_t* bitset_ptrs[kMaxRanks] = {nullptr};
            int n_ranks = dcp_size < kMaxRanks ? dcp_size : kMaxRanks;
            for (int r = 0; r < n_ranks; ++r) {
                const int gp = tp_gpus[r].position;
                if (gp < 0 || static_cast<size_t>(gp) >= moe_scratch_.size())
                    continue;
                bitset_ptrs[r] = moe_scratch_[gp].expert_resident_bitset.data();
            }
            // INV-MOE-EP-XTP: extra (expert-only) ranks join the dedup AFTER
            // the TP ranks — lowest-rank ownership keeps a duplicated expert
            // on a TP rank, deterministically.
            for (int g : active_extras) {
                if (n_ranks >= kMaxRanks) break;
                bitset_ptrs[n_ranks++] =
                    moe_scratch_[g].expert_resident_bitset.data();
            }
            const int dup_count = dedup_ep_residency(bitset_ptrs, n_ranks, n_experts);
            static const bool dup_dump = std::getenv("LS_EP_DUP_DUMP") != nullptr;
            if (dup_dump) {
                static std::atomic<uint64_t> dcall{0};
                spdlog::warn("LS_EP_DUP call={} layer={} cross_gpu_dup_experts={} (deduped)",
                             dcall.fetch_add(1), mp_template.layer_idx, dup_count);
            }
            // INV-MOE-EP-XTP: an extra rank may have owned only duplicates —
            // drop the now-empty ones (no dispatch, no fold, no stale rows).
            // INV-MOE-OVERLAP: keep a waved extra (accumulated rows pending).
            if (!active_extras.empty()) {
                active_extras.erase(
                    std::remove_if(active_extras.begin(), active_extras.end(),
                        [&](int g) {
                            if (mp_template.wave_pass == MoeWavePass::kFinal
                                && static_cast<size_t>(g)
                                       < moe_wave_accum_used_.size()
                                && moe_wave_accum_used_[g])
                                return false;
                            const auto& bs =
                                moe_scratch_[g].expert_resident_bitset;
                            for (size_t j = 0; j < bitset_bytes; ++j)
                                if (bs[j]) return false;
                            return true;
                        }),
                    active_extras.end());
            }
        }
    }

    // TD-PREFILL-MOE-BIG: batches beyond the single-shot bound run the chunked
    // pipeline per rank (dispatch_moe_chunked_internal, via
    // dispatch_moe_internal). Two cross-rank consequences here:
    //   - chunked DENSE layers write their FFN output into moe_output
    //     (INV-MOE-BIG-4) — the Phase-2 dense allreduce must read it;
    //   - the canonical per-slot EP combine is single-shot only
    //     (INV-MOE-BIG-3) — chunked routed batches use the legacy mode-0
    //     [B, H] bf16 combine.
    const bool chunked = deps_.cuda_kernels_enabled
        && num_tokens > moe_chunk_capacity_;

    // DET-REDUCE Phase 1b: use the placement-invariant canonical EP combine only
    // for the EP-within-TP routed path (the only placement-dependent cross-GPU
    // sum; shared/dense TP partitions are fixed). Requires the per-slot scratch
    // (allocated iff deterministic_ep_combine_). Payload precision = bf16 (mode 2)
    // when ep_combine_bf16_payload_, else fp32 (mode 1).
    const bool use_canonical_ep_combine =
        deterministic_ep_combine_ && ep_within_tp && !is_dense && !chunked;
    const uint8_t ep_combine_mode =
        ep_combine_bf16_payload_ ? 2 : 1;

    // ── C-6 early-kick reconciliation ────────────────────────────────────────
    // If the host FFN was early-kicked at fetch-launch (handle_fetch_and_run) for
    // THIS layer, it produced cpu_fold_moe_o_ in a PREDICTED format. If that
    // disagrees with this finalize's actual (perslot=use_canonical_ep_combine,
    // bf16=ep_combine_bf16_payload_) — a rare replicated-EP fallback — DISCARD the
    // early worker so fold_cpu_forced_experts re-produces synchronously with the
    // correct format (join_cpu_fold_worker clears cpu_fold_worker_kicked_). When it
    // agrees (the common sharded-EP path), the kick stands and the finalize's own
    // Phase-1 late kick below is SKIPPED so exactly one fold joins one worker.
    bool early_kick_valid = (mp_template.phase == MoeDispatchPhase::kFull
                             && cpu_early_kick_layer_ == mp_template.layer_idx);
    if (early_kick_valid
        && (cpu_kick_perslot_ != use_canonical_ep_combine
            || cpu_kick_bf16_ != ep_combine_bf16_payload_)) {
        spdlog::warn("C-6 early-kick: predicted fold format (perslot={}, bf16={}) "
                     "!= finalize (perslot={}, bf16={}) at layer {} — discarding "
                     "the early worker, re-producing synchronously",
                     cpu_kick_perslot_, cpu_kick_bf16_, use_canonical_ep_combine,
                     ep_combine_bf16_payload_, mp_template.layer_idx);
        join_cpu_fold_worker();
        early_kick_valid = false;
    }
    if (mp_template.phase == MoeDispatchPhase::kFull)
        cpu_early_kick_layer_ = 0xffffffffu;   // consumed this finalize

    // INV-MOE-EP-XTP: chunked batches cannot fold extra-rank partials (the
    // chunked pipeline accumulates into moe_output per chunk with row offsets;
    // the single-shot fold below does not compose with it). Fail loud rather
    // than silently drop the extra ranks' experts (TD-MOE-EP-XTP-WAVES).
    if (!active_extras.empty() && chunked) {
        spdlog::error("dispatch_moe_all_ranks: chunked MoE batch ({} tokens) "
                      "with {} expert-only rank(s) resident — EP-beyond-TP is "
                      "single-shot only (TD-MOE-EP-XTP-WAVES)",
                      num_tokens, active_extras.size());
        return false;
    }

    // INV-MOE-EP-XTP: under precomputed gating the routed top-K already sits
    // in rank0's device scratch — broadcast hidden+gating to the extra ranks
    // BEFORE the TP Phase-1 enqueue so their compute runs in parallel with
    // the TP ranks' (their only GPU dependency is attention + one rmsnorm).
    // Self-gating (legacy RUN_MOE) computes the top-K inside rank0's Phase-1
    // dispatch, so the broadcast must wait until after the TP loop.
    const bool xtp_pre_broadcast = mp_template.use_precomputed_gating;
    // INV-MOE-OVERLAP: the resident-overlap pass already broadcast this
    // layer's hidden + top-K to the extras — re-broadcasting would race the
    // extras' enqueued kPartial reads (rank0-stream D2D writes are not
    // ordered against the extras' kExpertFfn streams) for identical data.
    const bool xtp_already_broadcast =
        progressive_moe_.has_value() && progressive_moe_->xtp_broadcast_done
        && progressive_moe_->layer_idx == mp_template.layer_idx;
    if (!active_extras.empty() && xtp_pre_broadcast && !xtp_already_broadcast) {
        if (!ep_xtp_broadcast(mp_template, active_extras,
                              /*after_rank0_dispatch=*/false))
            return false;
    }

    // Phase 1: enqueue pre-allreduce MoE on all TP ranks.
    const uint32_t seg_key = mp_template.layer_idx << 16;  // §12h seam trace
    for (int r = 0; r < dcp_size; ++r) {
        InternalMoeParams mp = mp_template;
        mp.gpu_idx = static_cast<uint32_t>(tp_gpus[r].position);
        mp.phase = MoeDispatchPhase::kPreAllreduce;
        mp.bitset_precomputed = bitsets_precomputed;
        // Select the per-slot combine mode only if the matching scratch buffer for
        // this GPU is actually allocated (else fall back to the legacy bf16 sum).
        const auto& sr = moe_scratch_[tp_gpus[r].position];
        const bool buf_ready = ep_combine_bf16_payload_
            ? (sr.moe_output_bf16_perslot != nullptr)
            : (sr.moe_output_fp32 != nullptr);
        mp.ep_combine_mode =
            (use_canonical_ep_combine && buf_ready) ? ep_combine_mode : 0;
        perf_trace::record(perf_trace::kMoeSegRankPre,
                           static_cast<uint16_t>(mp.gpu_idx), 0, seg_key, 0);
        if (!dispatch_moe_internal(mp))
            return false;
        perf_trace::record(perf_trace::kMoeSegRankPre,
                           static_cast<uint16_t>(mp.gpu_idx), 0, seg_key, 1);
    }

    // ── C-6 Milestone C: KICK the host CPU-expert FFN NOW ───────────────────
    // Phase 1's per-rank FFN GEMMs are enqueued (async, in flight) and rank0's
    // norm+router + input-ready event are recorded. Launch the host FFN on a
    // worker thread so it runs CONCURRENTLY with the in-flight GPU GEMMs + the
    // Phase-2 EP allreduce below, instead of serializing behind them at the
    // finalize fold. fold_cpu_forced_experts (below) joins it and does only the
    // fast H2D fold. Kick params MUST match the fold that fires: perslot =
    // use_canonical_ep_combine, bf16 = ep_combine_bf16_payload_. Gated to kFull
    // + forced so exactly one fold joins it; no-op when overlap is OFF.
    // C-6 early-kick: when the host FFN was already kicked at fetch time
    // (early_kick_valid), SKIP this late kick — the fold below joins that worker,
    // which has been overlapping the fetch window since issue_moe_wave.
    if (mp_template.phase == MoeDispatchPhase::kFull
        && cpu_layer_has_forced(mp_template.layer_idx)
        && !early_kick_valid) {
        std::vector<int> kick_positions;
        kick_positions.reserve(dcp_size);
        for (int r = 0; r < dcp_size; ++r)
            kick_positions.push_back(tp_gpus[r].position);
        start_cpu_forced_experts(mp_template.layer_idx, num_tokens,
                                 kick_positions,
                                 /*perslot=*/use_canonical_ep_combine,
                                 /*bf16_payload=*/ep_combine_bf16_payload_);
    }

    // INV-MOE-EP-XTP: dispatch the expert-only ranks' routed subsets and fold
    // their partials onto the TP ranks BEFORE the Phase-2 EP combine.
    if (!active_extras.empty()) {
        if (!xtp_pre_broadcast &&
            !ep_xtp_broadcast(mp_template, active_extras,
                              /*after_rank0_dispatch=*/true))
            return false;
        if (!dispatch_moe_ep_extras(
                mp_template, active_extras,
                use_canonical_ep_combine ? ep_combine_mode : 0))
            return false;
    }

    // Phase 2: allreduce the partial sums across all TP ranks.
    // Dense FFN → expert_output; MoE → shared_expert_output.
    static constexpr int kMaxTp = 8;
    if (dcp_size > kMaxTp) {
        spdlog::critical("dispatch_moe_all_ranks: dcp_size {} exceeds kMaxTp {}", dcp_size, kMaxTp);
        std::abort();
    }
    void* buffers[kMaxTp];
    void* streams[kMaxTp];
    for (int r = 0; r < dcp_size; ++r) {
        const int gpu_pos = tp_gpus[r].position;
        if (gpu_pos < 0 || static_cast<size_t>(gpu_pos) >= moe_scratch_.size())
            return false;
        const auto& s = moe_scratch_[gpu_pos];
        // INV-MOE-BIG-4: chunked dense output lives in moe_output.
        buffers[r] = is_dense ? (chunked ? s.moe_output : s.expert_output)
                              : s.shared_expert_output;
        streams[r] = deps_.stream_manager->stream(
            gpu_pos, compute::StreamId::kExpertFfn);
    }
    // TD-74m: validate all buffers/streams before NCCL collective.
    for (int r = 0; r < dcp_size; ++r) {
        if (!buffers[r] || !streams[r]) {
            spdlog::error("dispatch_moe_all_ranks: rank {} has null buffer/stream", r);
            return false;
        }
    }
    perf_trace::record(perf_trace::kMoeSegNccl, 0, 0,
                       mp_template.layer_idx << 16, 0);

    // INV-NCCL-FUSE (default ON, LS_NCCL_FUSE=0 disables): when the routed
    // EP combine will also run for this layer, its allreduce is issued in the
    // SAME nccl group as this Phase-2 shared-expert reduce (one aggregated
    // device launch per rank instead of two tiny latency-bound ones, one host
    // enqueue instead of two). Bit-identical to the two separate calls.
    static const bool nccl_fuse = [] {
        const char* v = std::getenv("LS_NCCL_FUSE");
        return !(v && v[0] == '0');
    }();
    const bool ep_combine_runs = ep_within_tp && !is_dense;
    bool combine_fused = false;
    if (nccl_fuse && ep_combine_runs) {
        void* combine_bufs[kMaxTp];
        bool bufs_ok = true;
        const bool bf16_payload = ep_combine_bf16_payload_;
        for (int r = 0; r < dcp_size; ++r) {
            const int gpu_pos = tp_gpus[r].position;
            const auto& s = moe_scratch_[gpu_pos];
            combine_bufs[r] = use_canonical_ep_combine
                ? (bf16_payload ? s.moe_output_bf16_perslot : s.moe_output_fp32)
                : s.moe_output;
            if (!combine_bufs[r]) { bufs_ok = false; break; }
        }
        if (bufs_ok) {
            const int topk_f = deps_.live_config->model.num_experts_per_tok;
            const bool b_fp32 = use_canonical_ep_combine && !bf16_payload;
            const int b_rows = use_canonical_ep_combine ? topk_f : 1;

            // INV-NCCL-GRAPH: decode (B==1) replays the fused combine through
            // captured per-rank graphs (one capture; buffers are fixed scratch
            // addresses, validated against the baked signature every step).
            bool replayed = false;
            if (nccl_graph_enabled() && num_tokens == 1
                && !moe_combine_graph_failed_) {
                const int hidden_f = deps_.live_config->model.hidden_size;
                bool sig_ok = true;
                if (moe_combine_graph_ && moe_combine_graph_->is_captured()) {
                    sig_ok = moe_combine_graph_fp32_ == b_fp32
                          && moe_combine_graph_rows_ == b_rows
                          && static_cast<int>(moe_combine_graph_bufs_a_.size())
                                 == dcp_size;
                    for (int r = 0; sig_ok && r < dcp_size; ++r) {
                        sig_ok = moe_combine_graph_bufs_a_[r] == buffers[r]
                              && moe_combine_graph_bufs_b_[r] == combine_bufs[r];
                    }
                } else {
                    // First eligible step: capture (records, does not execute)
                    // then replay below to actually run this step's combine.
                    std::vector<int> dev_ids(dcp_size);
                    std::vector<void*> comms(dcp_size), strms(dcp_size);
                    std::vector<std::vector<compute::NcclGroupGraphRunner::Op>>
                        ops(dcp_size);
                    for (int r = 0; r < dcp_size; ++r) {
                        dev_ids[r] = tp_gpus[r].id;
                        comms[r] = deps_.dcp_communicator->comm(r);
                        strms[r] = streams[r];
                        ops[r] = {
                            {buffers[r],
                             static_cast<size_t>(num_tokens) * hidden_f,
                             false},
                            {combine_bufs[r],
                             static_cast<size_t>(num_tokens) * b_rows * hidden_f,
                             b_fp32}};
                    }
                    moe_combine_graph_ =
                        std::make_unique<compute::NcclGroupGraphRunner>();
                    if (moe_combine_graph_->init(dev_ids, comms, strms, ops)) {
                        moe_combine_graph_bufs_a_.assign(
                            buffers, buffers + dcp_size);
                        moe_combine_graph_bufs_b_.assign(
                            combine_bufs, combine_bufs + dcp_size);
                        moe_combine_graph_fp32_ = b_fp32;
                        moe_combine_graph_rows_ = b_rows;
                        spdlog::warn("INV-NCCL-GRAPH: captured fused MoE "
                                     "combine graphs ({} ranks)", dcp_size);
                    } else {
                        moe_combine_graph_.reset();
                        moe_combine_graph_failed_ = true;
                        sig_ok = false;
                    }
                }
                if (moe_combine_graph_ && moe_combine_graph_->is_captured()
                    && sig_ok) {
                    std::vector<void*> strms(streams, streams + dcp_size);
                    moe_combine_graph_->replay(strms);
                    replayed = true;
                }
            }
            if (!replayed) {
                deps_.dcp_communicator->allreduce_hidden_fused(
                    buffers, combine_bufs, num_tokens, streams,
                    /*b_fp32=*/b_fp32,
                    /*b_rows_per_token=*/b_rows);
            }
            combine_fused = true;
        }
    }
    if (!combine_fused) {
        deps_.dcp_communicator->allreduce_hidden(
            buffers, num_tokens, streams);
    }

    if (std::getenv("LS_DEBUG_GATING") && !is_dense &&
        static_cast<int>(mp_template.layer_idx) == first_k_dense) {
        const int n_experts = deps_.live_config->model.n_routed_experts;
        for (int r = 0; r < dcp_size; ++r) {
            const int gpu_pos = tp_gpus[r].position;
            const auto& bs = moe_scratch_[gpu_pos].expert_resident_bitset;
            int resident = 0;
            for (int e = 0; e < n_experts; ++e)
                if ((bs[e/8] >> (e%8)) & 1) ++resident;
            spdlog::warn("LS_DEBUG_GATING L{} phase={} ep={} rank{} gpu{} resident={}/{}",
                mp_template.layer_idx, static_cast<int>(mp_template.phase),
                ep_within_tp, r, gpu_pos, resident, n_experts);
        }
    }

    // 13c-7: EP-within-TP — allreduce routed moe_output across TP GPUs.
    // Each GPU computed a partial sum from its own expert subset; the allreduce
    // combines them so all GPUs have the full routed output before the
    // shared-expert add and residual in Phase 3.
    if (ep_within_tp && !is_dense) {
        if (use_canonical_ep_combine) {
            // DET-REDUCE Phase 1b (canonical, placement-INVARIANT). Each GPU wrote
            // its K expert contributions into PER-SLOT rows [token, slot, hidden]
            // (non-resident slots = 0). The cross-GPU allreduce GATHERS slots (each
            // filled by exactly one GPU; SUM = gather since the absent slots are 0
            // and 0+x=x exactly in both fp32 and bf16), then a FIXED-order K-slot
            // reduce accumulates in fp32 and rounds to bf16 ONCE into moe_output so
            // Phase 3 (residual add + commit) is unchanged. The per-slot value
            // c_k = w_k*expert_out_k is GPU-independent and the final sum is in
            // fixed slot order ⇒ the combine is BIT-identical regardless of which
            // GPU holds each expert (invariance is from the fixed ORDER, not the
            // payload dtype). Payload precision is selectable: fp32 (canonical) or
            // bf16 (half the gather bytes; matches the vLLM/llama.cpp convention).
            const int topk = deps_.live_config->model.num_experts_per_tok;
            const int hidden = deps_.live_config->model.hidden_size;
            const bool bf16_payload = ep_combine_bf16_payload_;
            void* moe_buffers_perslot[kMaxTp];
            for (int r = 0; r < dcp_size; ++r) {
                const int gpu_pos = tp_gpus[r].position;
                moe_buffers_perslot[r] = bf16_payload
                    ? moe_scratch_[gpu_pos].moe_output_bf16_perslot
                    : moe_scratch_[gpu_pos].moe_output_fp32;
                if (!moe_buffers_perslot[r] || !streams[r]) {
                    spdlog::error("dispatch_moe_all_ranks: per-slot EP allreduce rank "
                                  "{} has null per-slot buffer/stream", r);
                    return false;
                }
            }
            // Per-slot allreduce: count = num_tokens*topk*hidden_size (SUM =
            // gather); rows_per_token=topk keeps the max_batch_size guard on the
            // token count, not the slot-expanded count. fp32 payload → fp32
            // collective; bf16 payload → bf16 collective (half the bytes).
            // INV-NCCL-FUSE: already issued inside the Phase-2 group above.
            if (!combine_fused)
                deps_.dcp_communicator->allreduce_hidden(
                    moe_buffers_perslot, num_tokens, streams,
                    /*fp32=*/!bf16_payload, /*rows_per_token=*/topk);
            // ── C-6: fold forced CPU experts PER-SLOT (bit-exact, pre-combine).
            // The forced experts are excluded from every GPU bitset ⇒ their
            // per-slot rows are 0 on every rank after the gather allreduce
            // above. Add their host-computed per-slot contribution c_k into
            // EVERY rank identically (0 + c_k = c_k exact) BEFORE the
            // fixed-order combine reduce, so they ride the SAME canonical fp32
            // reduce as the GPU experts ⇒ ON == all-GPU OFF bit-for-bit under
            // LS_CPU_EXPERT_LOSSLESS. Runs once at finalize (kFull).
            if (mp_template.phase == MoeDispatchPhase::kFull
                && cpu_layer_has_forced(mp_template.layer_idx)) {
                std::vector<int> gpu_positions;
                gpu_positions.reserve(dcp_size);
                for (int r = 0; r < dcp_size; ++r)
                    gpu_positions.push_back(tp_gpus[r].position);
                if (!fold_cpu_forced_experts(mp_template.layer_idx, num_tokens,
                                             gpu_positions, /*perslot=*/true,
                                             /*bf16_payload=*/bf16_payload))
                    return false;
            }
            for (int r = 0; r < dcp_size; ++r) {
                const int gpu_pos = tp_gpus[r].position;
                deps_.device_backends[gpu_pos]->set_device();
                if (bf16_payload)
                    compute::launch_moe_combine_reduce_slots_bf16_to_bf16(
                        moe_scratch_[gpu_pos].moe_output,
                        moe_scratch_[gpu_pos].moe_output_bf16_perslot,
                        num_tokens, topk, hidden, streams[r]);
                else
                    compute::launch_moe_combine_reduce_slots_fp32_to_bf16(
                        moe_scratch_[gpu_pos].moe_output,
                        moe_scratch_[gpu_pos].moe_output_fp32,
                        num_tokens, topk, hidden, streams[r]);
            }
        } else if (!combine_fused) {
            // INV-NCCL-FUSE: when fused, the mode-0 moe_output reduce already
            // rode in the Phase-2 group above — nothing left to do here.
            void* moe_buffers[kMaxTp];
            for (int r = 0; r < dcp_size; ++r) {
                const int gpu_pos = tp_gpus[r].position;
                moe_buffers[r] = moe_scratch_[gpu_pos].moe_output;
            }
            for (int r = 0; r < dcp_size; ++r) {
                if (!moe_buffers[r] || !streams[r]) {
                    spdlog::error("dispatch_moe_all_ranks: EP allreduce rank {} "
                                  "has null moe_output/stream", r);
                    return false;
                }
            }
            deps_.dcp_communicator->allreduce_hidden(
                moe_buffers, num_tokens, streams);
        }
    }

    perf_trace::record(perf_trace::kMoeSegNccl, 0, 0,
                       mp_template.layer_idx << 16, 1);

    // ── C-6 Milestone A: legacy reduced-bf16 fold (mode-0 / non-canonical) ───
    // For the canonical per-slot EP combine the forced experts were already
    // folded PER-SLOT (bit-exact) before the combine reduce above — skip here.
    // This path covers the mode-0 [B,H] bf16 combine and the non-EP-within-TP
    // (replicated) case: moe_output now holds the GPU routed sum with the forced
    // experts' slots zero; add their host-reduced bf16 contribution onto EVERY
    // rank's moe_output (identical value ⇒ TP replicas stay in sync) on each
    // rank's kExpertFfn stream — after the combine, before Phase 3. Not
    // bit-exact (double bf16 rounding); kept only for the fallback modes.
    if (mp_template.phase == MoeDispatchPhase::kFull && !is_dense
        && !use_canonical_ep_combine
        && cpu_layer_has_forced(mp_template.layer_idx)) {
        std::vector<int> gpu_positions;
        gpu_positions.reserve(dcp_size);
        for (int r = 0; r < dcp_size; ++r)
            gpu_positions.push_back(tp_gpus[r].position);
        if (!fold_cpu_forced_experts(mp_template.layer_idx, num_tokens,
                                     gpu_positions))
            return false;
    }

    // TD-FAR-MULTICOMMIT: honor the incoming phase, symmetric to
    // dispatch_moe_internal. An incremental kPreAllreduce request runs Phase 1
    // (per-rank GEMMs) + Phase 2 (the TP allreduce above) but must NOT do the
    // Phase-3 residual add or the hidden-state commit — otherwise the progressive
    // FETCH_AND_RUN path (which dispatches once per arrival) would add the MoE
    // residual N times instead of once, corrupting the hidden state. Only kFull
    // (the finalize pass) and kPostAllreduce commit Phase 3, exactly once.
    if (mp_template.phase == MoeDispatchPhase::kPreAllreduce)
        return true;

    // Phase 3: finish residual add + commit on all TP ranks.
    for (int r = 0; r < dcp_size; ++r) {
        InternalMoeParams mp = mp_template;
        mp.gpu_idx = static_cast<uint32_t>(tp_gpus[r].position);
        mp.phase = MoeDispatchPhase::kPostAllreduce;
        perf_trace::record(perf_trace::kMoeSegRankPost,
                           static_cast<uint16_t>(mp.gpu_idx), 0, seg_key, 0);
        if (!dispatch_moe_internal(mp))
            return false;
        perf_trace::record(perf_trace::kMoeSegRankPost,
                           static_cast<uint16_t>(mp.gpu_idx), 0, seg_key, 1);
    }

    return true;
}

// ── INV-MOE-EP-XTP: EP degree beyond the DCP(TP) group ──────────────────────
// Expert-only GPUs (an ExpertDevice + expert cache, no attention/DCP rank)
// participate in the routed EP split: their resident subsets are computed
// in place and their per-token partials are D2D-folded onto a TP rank BEFORE
// the cross-rank EP combine, so the existing dcp_size-rank NCCL collective
// (mode-0 bf16 sum or the canonical per-slot gather) needs no new group.
// The fold is bit-exact under the per-slot modes (disjoint expert ownership
// enforced by dedup_ep_residency across TP+extra ranks ⇒ x + 0 = x).

bool CommandDispatcher::ep_xtp_broadcast(const InternalMoeParams& mp_template,
                                         const std::vector<int>& extras,
                                         bool after_rank0_dispatch) {
    if (!deps_.cuda_kernels_enabled || !deps_.stream_manager
        || !deps_.dcp_executor || !deps_.live_config)
        return false;
    const auto& tp_gpus = deps_.dcp_executor->gpus();
    if (tp_gpus.empty()) return false;
    const int src = tp_gpus[0].position;
    if (src < 0 || static_cast<size_t>(src) >= moe_scratch_.size()
        || static_cast<size_t>(src) >= deps_.device_backends.size()
        || !deps_.device_backends[src])
        return false;

    const auto& mc = deps_.live_config->model;
    const int num_tokens = static_cast<int>(mp_template.num_seqs);
    const int hidden = mc.hidden_size;
    const int topk = (mp_template.topk_override > 0)
        ? mp_template.topk_override : mc.num_experts_per_tok;
    const auto& ss = moe_scratch_[src];

    // Resolve rank0's raw hidden + post-attention norm weight (same lookups
    // dispatch_moe_internal makes for gpu==src).
    const int pair0 = resolve_pair_idx(static_cast<uint32_t>(src));
    void* raw_hidden = nullptr;
    if (pair0 >= 0) {
        raw_hidden = deps_.hidden_state_pairs[pair0].moe_buf;
    } else if (static_cast<size_t>(src) < deps_.fused_moe_hidden_states.size()) {
        raw_hidden = deps_.fused_moe_hidden_states[src];
    }
    const void* norm_w = nullptr;
    const int layer = static_cast<int>(mp_template.layer_idx);
    if (pair0 >= 0
        && layer < static_cast<int>(deps_.per_layer_attn_weights.size())) {
        const int r = deps_.hidden_state_pairs[pair0].rank;
        if (r >= 0
            && r < static_cast<int>(deps_.per_layer_attn_weights[layer].size()))
            norm_w = deps_.per_layer_attn_weights[layer][r]
                         .post_attention_layernorm;
    }
    if (!raw_hidden) {
        spdlog::error("ep_xtp_broadcast: no hidden state on rank0 gpu {}", src);
        return false;
    }

    auto* be_src = deps_.device_backends[src];
    be_src->set_device();
    void* s_stream = deps_.stream_manager->stream(
        src, compute::StreamId::kExpertFfn);

    // The source of the broadcast hidden: rank0's NORMALIZED hidden when a
    // norm weight exists (extra ranks have no norm weights — they consume the
    // hidden as-is), else the raw hidden (matching what rank0 itself feeds
    // its router/permute in that case).
    const void* src_hidden = raw_hidden;
    if (norm_w && ss.normalized_hidden) {
        if (!after_rank0_dispatch) {
            // Pre-Phase-1: order behind attention (KD-R2) and produce the
            // normalized hidden ourselves; rank0's own Phase-1 dispatch
            // recomputes it idempotently on the same stream.
            if (pair0 >= 0 && deps_.hidden_state_pairs[pair0].attn_moe_event) {
                deps_.stream_manager->wait_event(
                    src, compute::StreamId::kExpertFfn,
                    deps_.hidden_state_pairs[pair0].attn_moe_event);
            }
            // V4-5b mHC: collapse the hc-stream residual first (rank0's
            // own Phase-1 dispatch recomputes both idempotently).
            const void* bcast_rms_src = raw_hidden;
            if (deps_.hc_streams > 1) {
                const auto& lw0 = deps_.per_layer_attn_weights[layer]
                                      [deps_.hidden_state_pairs[pair0].rank];
                if (!lw0.hc_ffn_fn || !ss.hc_x) {
                    spdlog::error("ep_xtp_broadcast: mHC active but hc_ffn "
                                  "weights/scratch missing (layer {})", layer);
                    return false;
                }
                compute::launch_mhc_pre(
                    ss.hc_x, ss.hc_post, ss.hc_comb, raw_hidden,
                    lw0.hc_ffn_fn, lw0.hc_ffn_scale, lw0.hc_ffn_base,
                    deps_.live_config->model.rms_norm_eps,
                    deps_.live_config->model.hc_eps, 2.0f,
                    deps_.live_config->model.hc_sinkhorn_iters,
                    num_tokens, deps_.hc_streams, hidden, s_stream);
                bcast_rms_src = ss.hc_x;
            }
            compute::launch_rmsnorm(
                ss.normalized_hidden, bcast_rms_src, norm_w,
                deps_.live_config->model.rms_norm_eps,
                num_tokens, hidden,
                compute::NormDtype::kBFloat16, s_stream);
        }
        src_hidden = ss.normalized_hidden;
    }

    // Enqueue the D2D broadcasts on rank0's kExpertFfn stream (UVA peer copy
    // via DeviceBackend::memcpy_async — INV-GPU-1), then fence every extra
    // rank's kExpertFfn stream on one event.
    const size_t hidden_bytes = static_cast<size_t>(num_tokens) * hidden * 2;
    const size_t topk_w_bytes =
        static_cast<size_t>(num_tokens) * topk * sizeof(float);
    const size_t topk_i_bytes =
        static_cast<size_t>(num_tokens) * topk * sizeof(int32_t);
    for (int g : extras) {
        if (g < 0 || static_cast<size_t>(g) >= moe_scratch_.size()
            || static_cast<size_t>(g) >= deps_.fused_moe_hidden_states.size()
            || !deps_.fused_moe_hidden_states[g]) {
            spdlog::error("ep_xtp_broadcast: extra gpu {} has no MoE hidden "
                          "buffer", g);
            return false;
        }
        const auto& sg = moe_scratch_[g];
        if (!sg.topk_weights || !sg.topk_indices) {
            spdlog::error("ep_xtp_broadcast: extra gpu {} scratch not ready", g);
            return false;
        }
        be_src->memcpy_async(deps_.fused_moe_hidden_states[g], src_hidden,
                             hidden_bytes, s_stream);
        be_src->memcpy_async(sg.topk_weights, ss.topk_weights,
                             topk_w_bytes, s_stream);
        be_src->memcpy_async(sg.topk_indices, ss.topk_indices,
                             topk_i_bytes, s_stream);
    }
    void* ev = create_and_record_event(src, compute::StreamId::kExpertFfn);
    if (!ev) {
        spdlog::error("ep_xtp_broadcast: event create failed on gpu {}", src);
        return false;
    }
    for (int g : extras)
        deps_.stream_manager->wait_event(g, compute::StreamId::kExpertFfn, ev);
    // The driver retains the dependency for the enqueued waits.
    deps_.stream_manager->destroy_event(ev, src);
    return true;
}

bool CommandDispatcher::dispatch_moe_ep_extras(
        const InternalMoeParams& mp_template, const std::vector<int>& extras,
        uint8_t ep_combine_mode) {
    if (!deps_.cuda_kernels_enabled || !deps_.stream_manager
        || !deps_.dcp_executor || !deps_.live_config)
        return false;
    const auto& tp_gpus = deps_.dcp_executor->gpus();
    const int dcp_size = deps_.dcp_executor->dcp_size();
    if (tp_gpus.empty() || dcp_size < 1) return false;

    const auto& mc = deps_.live_config->model;
    const int num_tokens = static_cast<int>(mp_template.num_seqs);
    const int hidden = mc.hidden_size;
    const int topk = (mp_template.topk_override > 0)
        ? mp_template.topk_override : mc.num_experts_per_tok;

    // Mode-consistency: every rank of the combine (TP and extra) must write
    // the SAME payload kind, or the fold/collective would mix layouts.
    auto buf_for_mode = [&](const MoeScratch& s) -> void* {
        return ep_combine_mode == 2 ? s.moe_output_bf16_perslot
             : ep_combine_mode == 1 ? s.moe_output_fp32
                                    : s.moe_output;
    };

    // Phase 1b: dispatch each extra rank's resident subset (routed pipeline
    // only — kPreAllreduce + no shared-expert weights on expert-only GPUs ⇒
    // no residual, no commit).
    for (int g : extras) {
        if (g < 0 || static_cast<size_t>(g) >= moe_scratch_.size()) continue;
        if (!buf_for_mode(moe_scratch_[g])) {
            spdlog::error("dispatch_moe_ep_extras: extra gpu {} lacks the "
                          "mode-{} combine buffer", g, ep_combine_mode);
            return false;
        }
        InternalMoeParams mp = mp_template;
        mp.gpu_idx = static_cast<uint32_t>(g);
        mp.phase = MoeDispatchPhase::kPreAllreduce;
        mp.bitset_precomputed = true;   // built/deduped by the caller
        mp.use_precomputed_gating = true;  // top-K broadcast from rank0
        mp.ep_combine_mode = ep_combine_mode;
        // Prefill rolling waves stay TP-only (TD-MOE-EP-XTP-WAVES): an extra
        // rank that never wave-accumulated dispatches kNone (direct unpermute).
        // INV-MOE-OVERLAP: an extra that ran the decode resident-overlap
        // kPartial pass (accumulator touched) must dispatch kFinal so its
        // final rows are added and the unpermute reads the accumulator.
        const bool extra_waved = mp_template.wave_pass == MoeWavePass::kFinal
            && static_cast<size_t>(g) < moe_wave_accum_used_.size()
            && moe_wave_accum_used_[g];
        mp.wave_pass = extra_waved ? MoeWavePass::kFinal : MoeWavePass::kNone;
        mp.chunk_tokens = 0;
        mp.store_gating = false;  // rank0 already published this layer's routing
        perf_trace::record(perf_trace::kMoeSegExtra, static_cast<uint16_t>(g),
                           0, mp_template.layer_idx << 16, 0);
        if (!dispatch_moe_internal(mp)) {
            spdlog::error("dispatch_moe_ep_extras: dispatch failed on extra "
                          "gpu {} (layer {})", g, mp_template.layer_idx);
            return false;
        }
        perf_trace::record(perf_trace::kMoeSegExtra, static_cast<uint16_t>(g),
                           0, mp_template.layer_idx << 16, 1);
    }

    // Phase 1c: fold each extra rank's partial onto a TP rank. Payload:
    //   mode 0 → moe_output            [B, H]      bf16 (partial routed sum)
    //   mode 1 → moe_output_fp32       [B, K, H]   fp32 (per-slot)
    //   mode 2 → moe_output_bf16_perslot [B, K, H] bf16 (per-slot)
    // Copies run on the extra's kExpertFfn stream (ordered after its GEMMs);
    // the destination rank's kExpertFfn waits the copy, adds the staging into
    // its own same-mode buffer, and the later Phase-2 collective (enqueued on
    // the same stream) sees the folded partial. Consecutive extras hitting
    // the SAME destination rank are event-chained (the next copy must not
    // overwrite the staging before the previous add consumed it).
    const int n_elems = (ep_combine_mode == 0)
        ? num_tokens * hidden
        : num_tokens * topk * hidden;
    const size_t payload_bytes = static_cast<size_t>(n_elems)
        * (ep_combine_mode == 1 ? 4u : 2u);
    // §12h Variant-A diagnostic (LS_MOE_FOLD_VIA_HOST=1, DEFAULT OFF): route
    // the fold payload through pinned HOST RAM (D2H on the extra's stream →
    // cross-device event → H2D on the dst stream) instead of the direct D2D,
    // to measure the "CPU as concentrator" hop cost. Fully stream-ordered —
    // no host detection is added; this isolates the pure extra-hop latency.
    static const bool fold_via_host = [] {
        const char* v = std::getenv("LS_MOE_FOLD_VIA_HOST");
        return (v && v[0] == '1');
    }();
    perf_trace::record(perf_trace::kMoeSegFold, 0, 0,
                       mp_template.layer_idx << 16, 0);
    std::vector<void*> chain_ev(static_cast<size_t>(dcp_size), nullptr);
    for (size_t i = 0; i < extras.size(); ++i) {
        const int g = extras[i];
        if (g < 0 || static_cast<size_t>(g) >= moe_scratch_.size()
            || static_cast<size_t>(g) >= deps_.device_backends.size()
            || !deps_.device_backends[g])
            continue;
        const int r = static_cast<int>(i) % dcp_size;
        const int dst = tp_gpus[static_cast<size_t>(r)].position;
        if (dst < 0 || static_cast<size_t>(dst) >= moe_scratch_.size()
            || static_cast<size_t>(dst) >= deps_.device_backends.size()
            || !deps_.device_backends[dst])
            return false;
        auto& sd = moe_scratch_[dst];
        void* dst_payload = buf_for_mode(sd);
        if (!sd.ep_xtp_staging || !dst_payload) {
            spdlog::error("dispatch_moe_ep_extras: TP rank gpu {} lacks "
                          "staging/mode-{} buffer", dst, ep_combine_mode);
            return false;
        }

        // Extra's copy into dst staging (after any pending fold on dst).
        if (chain_ev[static_cast<size_t>(r)])
            deps_.stream_manager->wait_event(g, compute::StreamId::kExpertFfn,
                                             chain_ev[static_cast<size_t>(r)]);
        auto* be_g = deps_.device_backends[g];
        be_g->set_device();
        void* g_stream = deps_.stream_manager->stream(
            g, compute::StreamId::kExpertFfn);
        if (fold_via_host) {
            // Pinned bounce buffer per extra rank (lazy, grown on demand).
            if (fold_host_staging_.size() < deps_.device_backends.size())
                fold_host_staging_.resize(deps_.device_backends.size(),
                                          {nullptr, 0});
            auto& hs = fold_host_staging_[static_cast<size_t>(g)];
            if (hs.second < payload_bytes) {
                if (hs.first) be_g->host_free_pinned(hs.first);
                hs.first = be_g->host_alloc_pinned(payload_bytes);
                hs.second = hs.first ? payload_bytes : 0;
            }
            if (!hs.first) {
                spdlog::error("dispatch_moe_ep_extras: pinned fold staging "
                              "alloc failed (gpu {}, {} B)", g, payload_bytes);
                return false;
            }
            // Guard: the previous layer's H2D (dst stream) must have consumed
            // this host buffer before we overwrite it.
            if (fold_host_ev_.size() < deps_.device_backends.size())
                fold_host_ev_.resize(deps_.device_backends.size(),
                                     {nullptr, -1});
            auto& [guard_ev, guard_owner] =
                fold_host_ev_[static_cast<size_t>(g)];
            if (guard_ev) {
                deps_.stream_manager->wait_event(
                    g, compute::StreamId::kExpertFfn, guard_ev);
                deps_.stream_manager->destroy_event(guard_ev, guard_owner);
                guard_ev = nullptr;
                guard_owner = -1;
            }
            be_g->memcpy_d2h_async(hs.first, buf_for_mode(moe_scratch_[g]),
                                   payload_bytes, g_stream);
            void* d2h_ev = create_and_record_event(
                g, compute::StreamId::kExpertFfn);
            if (!d2h_ev) return false;
            deps_.stream_manager->wait_event(dst, compute::StreamId::kExpertFfn,
                                             d2h_ev);
            deps_.stream_manager->destroy_event(d2h_ev, g);
            auto* be_dst_h = deps_.device_backends[dst];
            be_dst_h->set_device();
            be_dst_h->memcpy_h2d_async(
                sd.ep_xtp_staging, hs.first, payload_bytes,
                deps_.stream_manager->stream(dst,
                                             compute::StreamId::kExpertFfn));
            fold_host_ev_[static_cast<size_t>(g)] = {
                create_and_record_event(dst, compute::StreamId::kExpertFfn),
                dst};
        } else {
        be_g->memcpy_async(sd.ep_xtp_staging, buf_for_mode(moe_scratch_[g]),
                           payload_bytes, g_stream);
        void* cp_ev = create_and_record_event(g, compute::StreamId::kExpertFfn);
        if (!cp_ev) return false;
        deps_.stream_manager->wait_event(dst, compute::StreamId::kExpertFfn,
                                         cp_ev);
        deps_.stream_manager->destroy_event(cp_ev, g);
        }

        // Fold on dst (bit-exact for per-slot modes: disjoint slots ⇒ x+0=x).
        auto* be_dst = deps_.device_backends[dst];
        be_dst->set_device();
        void* d_stream = deps_.stream_manager->stream(
            dst, compute::StreamId::kExpertFfn);
        if (ep_combine_mode == 1) {
            compute::launch_add_inplace_f32(dst_payload, sd.ep_xtp_staging,
                                            n_elems, d_stream);
        } else {
            compute::launch_residual_add(dst_payload, sd.ep_xtp_staging,
                                         n_elems, d_stream);
        }
        if (chain_ev[static_cast<size_t>(r)])
            deps_.stream_manager->destroy_event(
                chain_ev[static_cast<size_t>(r)], dst);
        chain_ev[static_cast<size_t>(r)] =
            create_and_record_event(dst, compute::StreamId::kExpertFfn);
    }
    for (int r = 0; r < dcp_size; ++r) {
        if (chain_ev[static_cast<size_t>(r)])
            deps_.stream_manager->destroy_event(
                chain_ev[static_cast<size_t>(r)],
                tp_gpus[static_cast<size_t>(r)].position);
    }
    perf_trace::record(perf_trace::kMoeSegFold, 0, 0,
                       mp_template.layer_idx << 16, 1);
    return true;
}

}  // namespace layerstorm::daemon
