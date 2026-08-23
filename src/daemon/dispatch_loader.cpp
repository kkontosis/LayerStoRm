// I8 GPU-loader shadow/act integration for the FETCH_AND_RUN_MOE path, split out
// of dispatch_moe.cpp. Everything here is env-gated (LS_LOADER_*) and inactive on
// the production decode path; it implements:
//   - init_loader_from_env() / close_loader_dump() — config + dump lifecycle
//   - route_moe_by_loader()  — P1 shadow solve+log + P2 (LS_LOADER_ACT) reroute
//   - shadow_solve_and_log() — the per-layer solve, log line, and JSONL x-ray dump
//   - feed_expert_stats()    — recency feed (only consumed by the loader's evict_cum)
// Keeping it here keeps the hot-path dispatcher (dispatch_moe.cpp) free of the
// loader machinery; the methods are CommandDispatcher members (state via members).
#include "daemon/command_dispatcher.h"
#include "daemon/dispatch_detail.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/device_backend.h"
#include "core/gpu_loader/loader_affinity_place.h"
#include "core/gpu_loader/loader_place_hotness.h"
#include "core/gpu_loader/loader_place_sum.h"
#include "core/gpu_loader/loader_policy.h"
#include "core/memory/arena_placement.h"
#include "core/memory/numa_manager.h"
#include "core/memory/pinned_expert_arena.h"
#include "core/expert_device.h"
#include "core/memory/expert_cache.h"
#include "core/memory/eviction_policy.h"
#include "core/memory/numa_manager.h"
#include "core/statistics/expert_stats.h"

namespace layerstorm::daemon {

namespace {
// LS_LOADER_MACH_PROF timestamp: steady_clock ns. Only called when the profiler
// is enabled (the callers guard on mach_prof_.enabled), so the default hot path
// pays exactly one predicted branch per block.
inline uint64_t mach_prof_now() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

// LS_LOADER_MACH_PROF summary: one table at teardown — per-block total ms,
// call count, µs/call, and µs per MoE layer-visit (layer count = kTouch calls,
// the once-per-route_moe_by_loader block).
void CommandDispatcher::report_loader_mach_prof() const {
    if (!mach_prof_.enabled) return;
    static const char* kNames[LoaderMachProf::kNumBlocks] = {
        "touch_loop", "req_build", "place_gather", "reuse_cost", "evict_curve",
        "solve", "result_out", "reroute", "ensure_victims", "ensure_residents"};
    const uint64_t layers = mach_prof_.calls[LoaderMachProf::kTouch];
    uint64_t total_ns = 0;
    for (int b = 0; b < LoaderMachProf::kNumBlocks; ++b)
        total_ns += mach_prof_.ns[b];
    spdlog::warn("LS_LOADER_MACH_PROF: {} layer-visits, machinery total "
                 "{:.2f} ms ({:.2f} us/layer)",
                 layers, total_ns / 1e6,
                 layers ? total_ns / 1e3 / static_cast<double>(layers) : 0.0);
    for (int b = 0; b < LoaderMachProf::kNumBlocks; ++b) {
        const uint64_t c = mach_prof_.calls[b];
        spdlog::warn("LS_LOADER_MACH_PROF:   {:<16} total={:.3f} ms  calls={}  "
                     "{:.3f} us/call  {:.3f} us/layer",
                     kNames[b], mach_prof_.ns[b] / 1e6, c,
                     c ? mach_prof_.ns[b] / 1e3 / static_cast<double>(c) : 0.0,
                     layers ? mach_prof_.ns[b] / 1e3 /
                                  static_cast<double>(layers)
                            : 0.0);
    }
    for (int f = 0; f < 10; ++f) {
        const uint64_t c = mach_prof_.solve_calls_by_free[f];
        if (!c) continue;
        spdlog::warn("LS_LOADER_MACH_PROF:   solve[nfree={}{}] total={:.3f} ms  "
                     "calls={}  {:.3f} us/call",
                     f, f == 9 ? "+" : "",
                     mach_prof_.solve_ns_by_free[f] / 1e6, c,
                     mach_prof_.solve_ns_by_free[f] / 1e3 /
                         static_cast<double>(c));
    }
}

// Read the LS_LOADER_* env gates once at construction. Default (all unset) = the
// loader is fully inert: no shadow solve, no routing change, no dump, no feed.
void CommandDispatcher::init_loader_from_env() {
    // I8 P1 shadow-mode gate (read once): compute+log j[·] vs e%tp, no routing
    // change. "0" explicitly DISABLES (harnesses default it on via
    // setenv(no-overwrite) — an A/B needs a way to force it off).
    {
        const char* sh = std::getenv("LS_LOADER_SHADOW");
        loader_shadow_ = sh != nullptr && !(sh[0] == '0' && sh[1] == '\0');
    }
    loader_act_ = (std::getenv("LS_LOADER_ACT") != nullptr);  // P2: act on j[·]
    // §12h NUMA-route bridge: dump the (layer,expert)→host-NUMA-node map once
    // (first FETCH command = preload done) so the harness/orchestrator router
    // can place misses on GPUs with a fast link to each expert's pinned slot.
    if (const char* p = std::getenv("LS_ARENA_MAP_DUMP"); p && *p)
        arena_map_dump_path_ = p;
    if (loader_act_) loader_shadow_ = true;                    // acting requires the solve
    // I8 Stage-4: full acting path but forced-identity assignment (mechanism isolation).
    loader_force_identity_ = (std::getenv("LS_LOADER_FORCE_IDENTITY") != nullptr);
    if (loader_force_identity_) { loader_act_ = true; loader_shadow_ = true; }
    // I8b model-vs-reality x-ray: structured JSONL dump path (opened lazily on
    // first solve). Implies the shadow solve must run; do not require LS_LOADER_SHADOW
    // separately — a dump path on its own enables the shadow solve for the dump.
    if (const char* dp = std::getenv("LS_LOADER_SHADOW_DUMP"); dp && *dp) {
        loader_shadow_dump_path_ = dp;
        loader_shadow_ = true;  // dump requires the shadow solve
    }
    // I8 P2: per-victim eviction unit cost for the solver's evict_cum term. <=0 (or
    // unset) → 0, which shadow_solve_and_log resolves to the mean bank egress_us
    // (~a representative re-fetch cost for a displaced resident).
    if (const char* eu = std::getenv("LS_LOADER_EVICT_UNIT_US"); eu && *eu) {
        loader_evict_unit_us_ = std::atof(eu);
        if (loader_evict_unit_us_ < 0.0) loader_evict_unit_us_ = 0.0;
    }
    // I8: evict horizon-discount γ — DEFAULT 0 since the reuse place term landed:
    // the trajectory x-ray (2026-07-17) measured the recency-raw-driven evict term
    // at 99.7% of predicted-T (clock inflation, p50 16→139 ms/layer over a run,
    // and the trainer has no coefficient for it) — it drowned the objective. The
    // place reuse reward below prices victim protection with a BOUNDED age signal
    // instead. LS_LOADER_EVICT_WEIGHT restores the term for A/B archaeology.
    if (const char* ew = std::getenv("LS_LOADER_EVICT_WEIGHT"); ew && *ew) {
        loader_evict_weight_ = std::atof(ew);
        if (loader_evict_weight_ < 0.0) loader_evict_weight_ = 0.0;
    }
    // I8 policy weights (P-26 unified deployment fit). Precedence: explicit env
    // vars (diagnostic arms) > LS_LOADER_POLICY artifact > built-in defaults
    // (the member initializers — the P-25 epoch-2 plateau optimum 60/0.1/2000/
    // 300). A malformed artifact fails LOUD (throws out of the ctor): a fitted
    // artifact carrying garbage means a bad fit, never a silent fallback.
    //
    // The knobs:
    //  * place_cons cross-token reuse reward (TD-LOADER-ROUTING-CROSSTOKEN):
    //    per-device miss-placement cost w/(1 + age/tau), age = recency_now −
    //    cheapest effective score (bounded, inflation-free; a trivialized
    //    duplicate victim reads as infinitely old → free to displace).
    //    Sim-developed on the committed keeper trajectories
    //    (loader_offline_sim/trajectory_sim.py --sweep); w<=0 disables.
    //  * decayed-frequency eviction protection (policy lab,
    //    POLICY_LAB_RESULTS.md): per routed access f = f·e^{-d} + 1 (saturates
    //    ≈ 1/(1−e^{-d})), stamped as a recency bonus w·f on every resident
    //    copy. w is in board layer-visit clock units — d=0.1 w=60 ⇒ ≤ ~8.4
    //    tokens of extra protection for a max-hot expert. w<=0 disables.
    {
        gpu_loader::PolicyParams pol;  // built-in defaults == member defaults
        std::string psrc = "builtin-defaults";
        if (const char* pp = std::getenv("LS_LOADER_POLICY"); pp && *pp) {
            pol  = gpu_loader::load_policy_params(pp);  // throws on malformed
            psrc = std::string("artifact ") + pp;
            if (!pol.epoch.empty()) psrc += " [epoch " + pol.epoch + "]";
        }
        if (gpu_loader::apply_policy_env_overrides(pol)) psrc += " + env overrides";
        loader_reuse_w_    = pol.reuse_w;
        loader_reuse_tau_  = pol.reuse_tau;
        loader_freq_w_     = pol.freq_w;
        loader_freq_decay_ = pol.freq_decay;
        spdlog::info("I8 policy weights ({}): freq_w={} freq_decay={} "
                     "reuse_w={} reuse_tau={}",
                     psrc, loader_freq_w_, loader_freq_decay_, loader_reuse_w_,
                     loader_reuse_tau_);
    }
    // EXPERIMENT: recency-decay reuse score into the EvictScoreBoard (per-token
    // raw *= factor + re-touch-on-use). 0/unset = off; clamp to (0,1].
    if (const char* ed = std::getenv("LS_EVICT_DECAY"); ed && *ed) {
        loader_evict_decay_ = std::atof(ed);
        if (loader_evict_decay_ < 0.0) loader_evict_decay_ = 0.0;
        if (loader_evict_decay_ > 1.0) loader_evict_decay_ = 1.0;
    }
    // Hot-path solve levers (TD-LOADER-SHADOW-HOTPATH-COST). Pin-hits defaults ON
    // (the pinned assignment is what ACT executes anyway; "0" disables for
    // full-domain x-ray studies). The per-layer shadow log line defaults OFF.
    {
        const char* ph = std::getenv("LS_LOADER_PIN_HITS");
        loader_pin_hits_ = !(ph && ph[0] == '0' && ph[1] == '\0');
    }
    {
        const char* sl = std::getenv("LS_LOADER_SHADOW_LOG");
        loader_shadow_log_ = sl != nullptr && !(sl[0] == '0' && sl[1] == '\0');
    }
    // Hoisted per-access decay multiplier (was an exp() per layer-visit).
    // (freq_w/freq_decay themselves are set in the P-26 policy block above.)
    loader_freq_mult_ = std::exp(-loader_freq_decay_);
    // M2v2 exposed-wall placement objective params (config-scoped fit JSON,
    // exposed_model.py --form v2). Absolute or cwd-relative path; loaded
    // lazily at first solve (needs num_devices). Pinned-tier-only downstream.
    if (const char* m2 = std::getenv("LS_LOADER_M2"); m2 && *m2)
        loader_m2_path_ = m2;
    // Machinery profiler (per-block ns accumulators; summary at teardown).
    mach_prof_.enabled = (std::getenv("LS_LOADER_MACH_PROF") != nullptr);
    // P-25: solver-expressed affinity placement (existence proof). "0" disables.
    {
        const char* pa = std::getenv("LS_LOADER_PLACE_AFFINITY");
        loader_place_affinity_ = pa != nullptr && !(pa[0] == '0' && pa[1] == '\0');
    }
    // TASK-A2 phase 1: hotness-steered place_cons (default OFF ⇒ byte-identical).
    {
        const char* ph = std::getenv("LS_LOADER_PLACE_HOTNESS");
        loader_place_hotness_ = ph != nullptr && !(ph[0] == '0' && ph[1] == '\0');
        if (const char* w = std::getenv("LS_LOADER_PLACE_HOTNESS_W"); w && *w)
            loader_place_hotness_w_ = std::atof(w);
        if (const char* t = std::getenv("LS_LOADER_PLACE_HOTNESS_TAU"); t && *t) {
            loader_place_hotness_tau_ = std::atof(t);
            if (loader_place_hotness_tau_ < 0.0) loader_place_hotness_tau_ = 0.0;
        }
        if (const char* c = std::getenv("LS_LOADER_PLACE_HOTNESS_CLAMP"))
            loader_place_hotness_clamp_ = !(c[0] == '0' && c[1] == '\0');
        // TASK-1: freq-prior coldness source (M3 CSV, same table as placement).
        if (const char* fq = std::getenv("LS_LOADER_PLACE_HOTNESS_FREQ");
            fq && *fq) {
            try {
                auto tbl = memory::ArenaFreqTable::load(fq);
                double mx = 0.0;
                for (const auto& [k, v] : tbl.counts) {
                    loader_place_hotness_freq_[k] = static_cast<float>(v);
                    if (v > mx) mx = v;
                }
                loader_place_hotness_use_freq_ = !loader_place_hotness_freq_.empty();
                if (const char* fh = std::getenv("LS_LOADER_PLACE_HOTNESS_FHOT");
                    fh && *fh)
                    loader_place_hotness_fhot_ = std::atof(fh);
                if (loader_place_hotness_fhot_ <= 0.0) loader_place_hotness_fhot_ = mx;
            } catch (const std::exception& e) {
                spdlog::error("LS_LOADER_PLACE_HOTNESS_FREQ load failed: {} — "
                              "falling back to board-eff coldness", e.what());
            }
        }
        if (loader_place_hotness_)
            spdlog::warn("LS_LOADER_PLACE_HOTNESS ON: w={} tau={} clamp={} "
                         "coldness={} (fhot={}, {} freq keys)",
                         loader_place_hotness_w_, loader_place_hotness_tau_,
                         loader_place_hotness_clamp_,
                         loader_place_hotness_use_freq_ ? "M3-freq-prior" : "board-eff",
                         loader_place_hotness_fhot_,
                         loader_place_hotness_freq_.size());
    }
    // TASK-2: never-lose I8 CPU-solver bridge (LS_LOADER_CPU_SOLVER=1). Appends a
    // CPU device to the live solver K with a calibrated host-FFN ComputeCurve; the
    // solver offloads a MISS to CPU only when cheaper than the GPU fetch+compute it
    // displaces (else zero ⇒ champion). Default OFF ⇒ byte-identical.
    {
        const char* cs = std::getenv("LS_LOADER_CPU_SOLVER");
        loader_cpu_solver_ = cs != nullptr && !(cs[0] == '0' && cs[1] == '\0');
        // C-6 Milestone C: the solver's cost-model node tracks the device's
        // effective node. LS_LOADER_CPU_SOLVER_NODE (explicit) wins; else follow
        // LS_CPU_EXPERT_NODE (the physical placement, default 1) so the modeled
        // transfer/DDR bank matches where the host FFN + fold staging actually run.
        if (const char* nd = std::getenv("LS_LOADER_CPU_SOLVER_NODE"); nd && *nd)
            loader_cpu_solver_node_ = std::atoi(nd);
        else if (const char* cn = std::getenv("LS_CPU_EXPERT_NODE"); cn && *cn)
            loader_cpu_solver_node_ = std::atoi(cn);
        else
            loader_cpu_solver_node_ = 1;
        // C-6 Milestone C: skip the per-layer never-lose SOLVE on B=1 (plain
        // decode / draft) steps. At B=1 the solve is provably k=0 (the host FFN
        // a_fixed+a_pertok ≫ a single GPU fetch), so running it just adds daemon
        // overhead to the plain-step critical path (measured +21% B=1 plain-step
        // tax with ~zero offload). DEFAULT ON; =0 restores the every-step solve.
        loader_cpu_skip_b1_ = true;
        if (const char* sk = std::getenv("LS_LOADER_CPU_SKIP_B1");
            sk && sk[0] == '0' && sk[1] == '\0')
            loader_cpu_skip_b1_ = false;
        if (const char* a = std::getenv("LS_LOADER_CPU_A_US"); a && *a)
            loader_cpu_a_us_ = std::atof(a);
        if (const char* b = std::getenv("LS_LOADER_CPU_B_US"); b && *b)
            loader_cpu_b_us_ = std::atof(b);
        if (const char* pp = std::getenv("LS_LOADER_CPU_P"); pp && *pp)
            loader_cpu_p_ = std::max(1, std::atoi(pp));
        if (const char* m = std::getenv("LS_LOADER_CPU_COST_MULT"); m && *m) {
            loader_cpu_cost_mult_ = std::atof(m);
            if (loader_cpu_cost_mult_ < 0.0) loader_cpu_cost_mult_ = 0.0;
        }
        // TASK-1: B-aware host-FFN decomposition + overlap knobs. Defaults match
        // the offline B-sweep sim (a_fixed 907 + a_pertok 14 ⇒ 921 µs/tok at B=1,
        // 71 µs/tok at B=16; η floor 0.15, τ 4). Env-overridable for sensitivity.
        if (const char* af = std::getenv("LS_LOADER_CPU_A_FIXED_US"); af && *af)
            loader_cpu_a_fixed_us_ = std::atof(af);
        if (const char* ap = std::getenv("LS_LOADER_CPU_A_PERTOK_US"); ap && *ap)
            loader_cpu_a_pertok_us_ = std::atof(ap);
        if (const char* of = std::getenv("LS_LOADER_CPU_OVERLAP_FLOOR"); of && *of) {
            loader_cpu_overlap_floor_ = std::atof(of);
            if (loader_cpu_overlap_floor_ < 0.0) loader_cpu_overlap_floor_ = 0.0;
            if (loader_cpu_overlap_floor_ > 1.0) loader_cpu_overlap_floor_ = 1.0;
        }
        if (const char* ot = std::getenv("LS_LOADER_CPU_OVERLAP_TAU"); ot && *ot) {
            loader_cpu_overlap_tau_ = std::atof(ot);
            if (loader_cpu_overlap_tau_ < 1e-3) loader_cpu_overlap_tau_ = 1e-3;
        }
        if (loader_cpu_solver_)
            spdlog::warn("LS_LOADER_CPU_SOLVER ON: CPU device on node {} — B-AWARE "
                         "host FFN a_fixed={} a_pertok={} (per-expert@B = a_fixed+"
                         "a_pertok·B) fold b_us={} (x mult {}); overlap η(B)=1−(1−{})"
                         "·exp(−(B−1)/{}); LAYER-LEVEL never-lose (k=0 champion always "
                         "in search ⇒ never worse; B=1 ⇒ k*=0)",
                         loader_cpu_solver_node_,
                         loader_cpu_a_fixed_us_ * loader_cpu_cost_mult_,
                         loader_cpu_a_pertok_us_ * loader_cpu_cost_mult_,
                         loader_cpu_b_us_ * loader_cpu_cost_mult_,
                         loader_cpu_cost_mult_, loader_cpu_overlap_floor_,
                         loader_cpu_overlap_tau_);
    }
    // Residency reframe: weighted-sum place_cons for the never-lose CPU offload
    // (loader_place_sum.h). DEFAULT ON (=1); the CPU-column cost is a tunable
    // weighted sum of (a) residency/future-miss baseline (dominant), (b) NUMA
    // home-locality, (c) freq/EMA hotness. When ON, an offload pays its honest CPU
    // cost = exposed host FFN + place_cons; the never-lose greedy offloads ONLY
    // experts where that is below the GPU fetch+compute it displaces — so it stops
    // churning residency (would-recur HITs stay on GPU). =0 zeroes every weight ⇒
    // the raw pre-residency wall model (the naive greedy). Inert unless CPU offload
    // (LS_LOADER_CPU_SOLVER) is engaged ⇒ champion-safe / byte-identical when off.
    {
        const char* ps = std::getenv("LS_LOADER_PLACE_SUM");
        loader_place_sum_ = !(ps && ps[0] == '0' && ps[1] == '\0');  // default ON
        if (!loader_place_sum_) loader_place_sum_w_ = gpu_loader::PlaceSumWeights{0, 0, 0, 0};
        // resid_tax + w_resid + w_hotness are HEURISTICS (env-only, never trained);
        // only w_numa is adopted from the trained calibration JSON (below).
        if (const char* rt = std::getenv("LS_LOADER_PLACE_RESID_TAX"); rt && *rt)
            loader_place_sum_w_.resid_tax = std::atof(rt);
        if (const char* wr = std::getenv("LS_LOADER_PLACE_W_RESID"); wr && *wr)
            loader_place_sum_w_.w_resid = std::atof(wr);
        if (const char* wn = std::getenv("LS_LOADER_PLACE_W_NUMA"); wn && *wn) {
            loader_place_sum_w_.w_numa = std::atof(wn);
            loader_place_w_numa_env_ = true;
        }
        if (const char* wh = std::getenv("LS_LOADER_PLACE_W_HOTNESS"); wh && *wh)
            loader_place_sum_w_.w_hotness = std::atof(wh);
        // (d) NODE-BASED offload-decision bias (predictor-free; STATIC home-node fact).
        // HBM-home (M3-hot) → LOW base (encourage offload → free pinned VRAM for the
        // cold tail); DDR-home (cold tail) → HIGH penalty (protect the fetch-bound
        // tail). Both default 0 ⇒ node term inert (champion byte-identical).
        if (const char* hb = std::getenv("LS_LOADER_PLACE_HBM_OFFLOAD_BASE"); hb && *hb)
            loader_place_sum_w_.hbm_offload_base = std::atof(hb);
        if (const char* dp = std::getenv("LS_LOADER_PLACE_DDR_OFFLOAD_PENALTY"); dp && *dp)
            loader_place_sum_w_.ddr_offload_penalty = std::atof(dp);
        // (e) TARGET-NODE RARE offload steering (predictor-free: location_node fact
        // ANDed with M3 rarity). LS_LOADER_PLACE_OFFLOAD_NODE = the GPU-FREE DDR node
        // whose experts are the most expensive to GPU-fetch (default -1 ⇒ (e) off).
        // Sentinel -2 = ANY node (pure-rarity budget rung: target_rare drops the
        // node-membership AND — rare⇔normM3<thresh alone; fetch still priced from
        // the REAL holding node).
        // LS_LOADER_PLACE_RARITY_THRESH = normalized-M3-hotness cutoff (rare ⇔ h <
        // thresh; needs LS_LOADER_PLACE_HOTNESS_FREQ). offnode_protect PROTECTs the
        // complement, target_rare_encourage drives the target-rare set to the 0 floor.
        if (const char* on = std::getenv("LS_LOADER_PLACE_OFFLOAD_NODE"); on && *on)
            loader_place_offload_node_ = std::atoi(on);
        if (const char* rth = std::getenv("LS_LOADER_PLACE_RARITY_THRESH"); rth && *rth)
            loader_place_rarity_thresh_ = std::atof(rth);
        if (const char* op = std::getenv("LS_LOADER_PLACE_OFFNODE_PROTECT"); op && *op)
            loader_place_sum_w_.offnode_protect = std::atof(op);
        if (const char* te = std::getenv("LS_LOADER_PLACE_TARGET_RARE_ENCOURAGE"); te && *te)
            loader_place_sum_w_.target_rare_encourage = std::atof(te);
        if (loader_cpu_solver_)
            spdlog::warn("LS_LOADER_PLACE_SUM {}: CPU-column place_cons = resid_tax={}us "
                         "(HEURISTIC churn floor) + w_resid={}·recur·fetch (HEURISTIC) + "
                         "w_numa={} (TRAINED) + w_hotness={} (HEURISTIC) + node-bias["
                         "HBM+{}us / DDR+{}us] (STATIC home-node)",
                         loader_place_sum_ ? "ON" : "OFF(zeroed)",
                         loader_place_sum_w_.resid_tax, loader_place_sum_w_.w_resid,
                         loader_place_sum_w_.w_numa, loader_place_sum_w_.w_hotness,
                         loader_place_sum_w_.hbm_offload_base,
                         loader_place_sum_w_.ddr_offload_penalty);
        if (loader_cpu_solver_ && loader_place_offload_node_ != -1)
            spdlog::warn("LS_LOADER_PLACE (e) TARGET-NODE RARE offload: node={} "
                         "rarity_thresh={} (rare⇔normM3<thresh) offnode_protect={}us "
                         "target_rare_encourage={}us", loader_place_offload_node_,
                         loader_place_rarity_thresh_, loader_place_sum_w_.offnode_protect,
                         loader_place_sum_w_.target_rare_encourage);
    }
    // P-25: legacy-term ablation list. Defaults to the full legacy set under
    // place-affinity (the synthetic objective must not be perturbed by real
    // transfer/compute/bank/recon costs); explicit LS_LOADER_ABLATE overrides
    // (single-term walk-back arms).
    {
        const char* ab = std::getenv("LS_LOADER_ABLATE");
        std::string list = ab ? ab
                              : (loader_place_affinity_ ? "xfer,compute,bank,recon" : "");
        size_t p0 = 0;
        while (p0 < list.size()) {
            size_t p1 = list.find(',', p0);
            if (p1 == std::string::npos) p1 = list.size();
            const std::string tok = list.substr(p0, p1 - p0);
            if      (tok == "xfer")    loader_ablate_mask_ |= kAblateXfer;
            else if (tok == "compute") loader_ablate_mask_ |= kAblateCompute;
            else if (tok == "bank")    loader_ablate_mask_ |= kAblateBank;
            else if (tok == "recon")   loader_ablate_mask_ |= kAblateRecon;
            else if (!tok.empty())
                spdlog::warn("LS_LOADER_ABLATE: unknown term '{}' (want "
                             "xfer|compute|bank|recon)", tok);
            p0 = p1 + 1;
        }
    }
}

// P-25 place-affinity fine-tick mirror (see the member comment): stamp every
// routed expert at its EXECUTED device in the keeper model's application order
// — per-GPU grouped (g ascending), hits (board-resident at g) before misses,
// routed order within each group. Called once per layer with the final
// positions (fast path: orchestrator targets; solved path: post-canonicalize
// assignment). Ticks are strictly increasing ⇒ cross-device age comparisons
// are total, matching the keeper's g_lru_tick.
void CommandDispatcher::aff_fine_stamp(const ipc::ExpertPrefetchEntry* entries,
                                       uint32_t n, const int* pos_of_expert) {
    const size_t nb = deps_.device_backends.size();
    if (aff_fine_.size() != nb) {
        aff_fine_.resize(nb);
        aff_fine_heap_.resize(nb);
    }
    const auto cmp = [](const std::pair<uint64_t, memory::ExpertKey>& a,
                        const std::pair<uint64_t, memory::ExpertKey>& b) {
        return a.first > b.first;  // min-heap on tick
    };
    for (int g = 0; g < static_cast<int>(nb); ++g) {
        auto& heap = aff_fine_heap_[static_cast<size_t>(g)];
        auto& fine = aff_fine_[static_cast<size_t>(g)];
        for (int pass = 0; pass < 2; ++pass)  // 0 = hits, 1 = misses
            for (uint32_t i = 0; i < n; ++i) {
                if (pos_of_expert[i] != g) continue;
                const auto key = make_key(entries[i].layer_idx,
                                          entries[i].expert_idx);
                const bool hit = evict_board_ && evict_board_->is_resident(g, key);
                if ((pass == 0) != hit) continue;
                const uint64_t t = ++aff_fine_tick_;
                fine[key] = t;
                heap.emplace_back(t, key);
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
    }
}

// Oldest fine-stamped board-resident on `pos`, skipping the exclusion list
// (this layer's hits on the device — the keeper router's needed_now). Lazy
// invalidation: an entry is stale when the map holds a newer tick for its key
// (re-touch/re-fetch) or the key is no longer board-resident (evicted).
// Returns 1e18 ("no evictable victim — last resort") when nothing remains.
double CommandDispatcher::aff_fine_oldest(int pos, const memory::ExpertKey* excl,
                                          int nexcl) {
    if (pos < 0 || pos >= static_cast<int>(aff_fine_heap_.size())) return 1e18;
    auto& heap = aff_fine_heap_[static_cast<size_t>(pos)];
    auto& fine = aff_fine_[static_cast<size_t>(pos)];
    const auto cmp = [](const std::pair<uint64_t, memory::ExpertKey>& a,
                        const std::pair<uint64_t, memory::ExpertKey>& b) {
        return a.first > b.first;
    };
    std::array<std::pair<uint64_t, memory::ExpertKey>,
               gpu_loader::kMaxExperts> saved;
    int nsaved = 0;
    double res = 1e18;
    while (!heap.empty()) {
        const auto top = heap.front();
        const auto it = fine.find(top.second);
        const bool stale = it == fine.end() || it->second != top.first ||
                           !evict_board_ ||
                           !evict_board_->is_resident(pos, top.second);
        if (stale) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.pop_back();
            continue;
        }
        bool excluded = false;
        for (int e = 0; e < nexcl; ++e)
            if (excl[e] == top.second) { excluded = true; break; }
        if (excluded && nsaved < gpu_loader::kMaxExperts) {
            saved[static_cast<size_t>(nsaved++)] = top;
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.pop_back();
            continue;
        }
        res = static_cast<double>(top.first);
        break;
    }
    for (int s = 0; s < nsaved; ++s) {
        heap.push_back(saved[static_cast<size_t>(s)]);
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
    return res;
}

// P-25: lazily materialize the ablated LoaderConstants copy (rebuilt if the
// source pointer ever changes, e.g. a runtime recalibration swap).
const gpu_loader::LoaderConstants& CommandDispatcher::ablated_loader_constants() {
    const auto* src = deps_.loader_constants;
    if (loader_k_ablated_src_ != src) {
        loader_k_ablated_src_ = src;
        loader_k_ablated_ = *src;
        auto& k = loader_k_ablated_;
        if (loader_ablate_mask_ & kAblateXfer) {
            for (auto& row : k.matrix)
                for (auto& c : row) { c.rate_us = 0.0; c.lat_us = 0.0; }
            for (auto& d : k.devices) d.xfer_lat_us = 0.0;
        }
        if (loader_ablate_mask_ & kAblateCompute)
            for (auto& d : k.devices) d.compute = {};
        if (loader_ablate_mask_ & kAblateBank)
            for (auto& b : k.banks) b.egress_us = 0.0;
        if (loader_ablate_mask_ & kAblateRecon)
            for (auto& d : k.devices) {
                d.recon_overhead_us = 0.0;
                d.recon_added_us    = 0.0;
            }
        spdlog::info("LS_LOADER_ABLATE: solver constants ablated (mask=0x{:x})",
                     loader_ablate_mask_);
    }
    return loader_k_ablated_;
}

#ifndef NDEBUG
// Debug parity check (TD-EVICT-BOARD-DESYNC): the EvictScoreBoard's per-GPU
// residency must be EXACTLY ExpertCache's stable-zone residency. Since the board
// is now the cache's ResidencyListener, the only way these can diverge is a path
// that mutates stable residency without going through ExpertCache's add/evict
// choke-points — which this assert makes trip immediately. Debug-only (O(N), off
// the release hot path entirely).
void CommandDispatcher::assert_board_cache_parity() const {
    if (!evict_board_ || !deps_.expert_cache) return;
    const int M = evict_board_->num_gpus();
    std::vector<memory::ExpertKey> board_keys;
    for (int g = 0; g < M; ++g) {
        // Cache stable residents on g.
        std::unordered_set<memory::ExpertKey> cache_stable;
        for (const auto& ri : deps_.expert_cache->residency_snapshot(g))
            if (ri.zone == memory::CacheZone::kStable) cache_stable.insert(ri.key);
        // Board residents on g.
        evict_board_->resident_keys(g, board_keys);
        assert(board_keys.size() == cache_stable.size() &&
               "EvictScoreBoard residency size != ExpertCache stable residency");
        for (const auto& k : board_keys)
            assert(cache_stable.count(k) == 1 &&
                   "EvictScoreBoard holds a key absent from ExpertCache stable zone");
    }
}
#endif  // NDEBUG

void CommandDispatcher::close_loader_dump() {
    if (loader_shadow_dump_fp_) {
        std::fclose(loader_shadow_dump_fp_);
        loader_shadow_dump_fp_ = nullptr;
    }
}

// ── LOADER_STATS_LOCALITY: lazy build of the per-GPU evict-score board +
// place_cons table (spec/tickets/LOADER_STATS_LOCALITY.md). Built the first time
// the loader path runs; nullopt = inert (production decode path never gets here
// because every caller is loader_shadow_-gated). Sized from live_config so the
// flat expert id space (layer × expert) is a superset of every routed key (no
// OOB). M = the live device-backend count. ──────────────────────────────────
void CommandDispatcher::ensure_loader_stats_boards() {
    if (loader_stats_boards_built_) return;
    loader_stats_boards_built_ = true;  // build-once (also on the no-config path)
    if (!deps_.live_config) return;     // cannot size → leave inert
    const auto& m = deps_.live_config->model;
    const int num_layers = m.num_hidden_layers > 0 ? m.num_hidden_layers : 0;
    const int experts_per_layer = m.n_routed_experts > 0 ? m.n_routed_experts : 0;
    const int first_moe = m.first_k_dense_replace > 0 ? m.first_k_dense_replace : 0;
    const int M = static_cast<int>(deps_.device_backends.size());
    if (num_layers <= 0 || experts_per_layer <= 0 || M <= 0) return;

    loader_first_moe_layer_   = static_cast<uint32_t>(first_moe);
    loader_experts_per_layer_ = static_cast<uint32_t>(experts_per_layer);
    // num_moe_layers spans first_moe..num_layers (a superset; dense layers below
    // first_moe never route experts, so their slice is simply unused). Sizing to
    // the full layer range keeps every layer_idx in range with no freq tracking.
    loader_num_moe_layers_ = static_cast<uint32_t>(
        num_layers > first_moe ? (num_layers - first_moe) : num_layers);

    const int flat_experts =
        static_cast<int>(loader_num_moe_layers_) * experts_per_layer;
    place_table_.emplace(flat_experts, M);
    // cap_per_gpu: a GPU's stable-zone resident count is the natural cap; use the
    // expert-cache stable slot count when available, else a generous default.
    int cap = 1024;
    if (deps_.expert_cache && M > 0) {
        int s = deps_.expert_cache->total_slots(0, memory::CacheZone::kStable);
        if (s > 0) cap = s;
    }
    evict_board_.emplace(M, cap);
    // baseScoreValue (the per-rank duplicate trivialization penalty) is left at
    // the 0.9 default. Scores are driven by the always-on recency clock (raw =
    // recency on add / routed touch) plus the optional LS_EVICT_DECAY experiment.
    evict_board_->set_base_score(gpu_loader::kDefaultBaseScore);

    // TD-EVICT-BOARD-DESYNC: register the board as the ExpertCache residency
    // listener. From here the cache fires on_resident_added/removed at its own
    // stable-zone choke-points, so the board AUTHORITATIVELY mirrors stable
    // residency (no more scattered loader_on_place/loader_on_evict, no desync).
    // The board lives in this dispatcher's std::optional (emplaced in place, never
    // moved); it is unregistered in the dispatcher destructor.
    if (deps_.expert_cache) {
        // SEED the board with the cache's CURRENT stable residents — the board is
        // built lazily (first loader-gated solve), by which point prefill /
        // earlier prefetches may already hold stable slots the listener never saw.
        // Without this one-time sync the board would start short of the cache and
        // never reach parity. Seed at recency 0 (oldest); routed use re-touches.
        for (int g = 0; g < M; ++g)
            for (const auto& ri : deps_.expert_cache->residency_snapshot(g))
                if (ri.zone == memory::CacheZone::kStable)
                    evict_board_->on_resident_added(ri.key, g);
        deps_.expert_cache->set_residency_listener(&*evict_board_);
    }
}

int CommandDispatcher::loader_flat_expert_id(uint32_t layer_idx,
                                             uint16_t expert_idx) const {
    if (loader_experts_per_layer_ == 0) return -1;
    if (layer_idx < loader_first_moe_layer_) return -1;
    const uint32_t rel = layer_idx - loader_first_moe_layer_;
    if (rel >= loader_num_moe_layers_) return -1;
    if (expert_idx >= loader_experts_per_layer_) return -1;
    return static_cast<int>(rel * loader_experts_per_layer_ + expert_idx);
}

// LOADER_STATS_LOCALITY: advance the loader's per-TOKEN clock exactly once per
// token. A new token's MoE sequence restarts at the lowest MoE layer, so a
// STRICT DECREASE in layer_idx (or the very first layer) starts a token. Strict
// `<` (not `<=`) makes this IDEMPOTENT within a layer: it is called both at the
// top of route_moe_by_loader (so the solve sees the fresh clock) AND from
// feed_expert_stats (which runs even when loader_constants is absent and route
// early-returns) — the second call for the SAME layer has layer_idx ==
// last_stats_layer_, so it does NOT re-advance. Within one token the layer index
// strictly increases (3,4,5,…), so only the wrap-down to the lowest MoE layer
// bumps the clock. Updates stats_token_id_ / last_stats_layer_ — the single
// token-tracking state shared with the ExpertStats feed.
bool CommandDispatcher::advance_loader_token_if_new(uint32_t layer_idx) {
    const bool new_token =
        (last_stats_layer_ == UINT32_MAX || layer_idx < last_stats_layer_);
    if (new_token) {
        ++stats_token_id_;
        // EVICTBOARD_EXTERNAL_SCORES: the board no longer keeps an internal token
        // clock (scores are external). Only stats_token_id_ (the off-hot-path
        // ExpertStats feed's token id) still advances here.
    }
    last_stats_layer_ = layer_idx;
    return new_token;
}

// I8 P1 shadow-mode: solve expert→device with the loader cost model and LOG it
// beside the orchestrator's gpu_idx. P2 (LS_LOADER_ACT) additionally ROUTES by
// the solver's j[·] — but ONLY for miss experts (was_cached==false), so a cache
// hit is never turned into a miss (cache-safe). WARNING (TD-MOE-EP-COMBINE-FPDRIFT,
// I8 §9): the routed output is NOT bit-unchanged by the reroute. The per-EXPERT
// result is GPU-independent, but the per-TOKEN cross-GPU COMBINE is NOT: each GPU
// fp32-accumulates its resident subset's weighted outputs then rounds the PARTIAL
// to bf16 (finalize_moe_routing_bf16), and the EP allreduce adds the two bf16
// partials — so moving an expert to a different GPU shifts a partial's rounding
// boundary → a bf16-ULP MoE-output delta → near-tie gating flips → an autoregressive
// trajectory fork. This is a placement-dependent FP-reduction-order effect (a SECOND
// site after the attention denominator DET-REDUCE fixed); it is OUTPUT-token-robust
// (golden-green) but NOT numerically inert. PROVEN: FORCE_IDENTITY (canonical e%tp
// partition) collapses ACT+decay keeper 0.629→0.400 byte-identical to baseline.
// Inert (early return) unless the loader is enabled and calibrated.
void CommandDispatcher::route_moe_by_loader(
    const ipc::ExpertPrefetchEntry* entries, const ipc::Command& cmd,
    ProgressiveMoeState& state) {
    // INV-REEF-BANK first-epoch publish for CLIENT-side ReefOrch consumers
    // (the KEEPER52_REEF_ORCH test arm): the daemon service publishes at
    // its own construction, but a test-side stack solves on the client
    // thread and can only consume a snapshot PUBLISHED from the daemon
    // thread. This site (first routed-MoE fetch) is where the retired
    // boot-CSV export fired — same epoch-1 timing, byte-compatible race
    // semantics (the first solve pre-dates it and reads bank 0).
    if (deps_.pinned_arena && deps_.pinned_arena->bank_epoch() == 0)
        deps_.pinned_arena->publish_bank_snapshot();
    // §12h NUMA-route bridge: one-shot arena location-map dump (atomic rename
    // so a polling reader never sees a partial file). Paired node = the
    // CPU-affinity node for HBM banks (H2D rate class follows the pair).
    if (!arena_map_dump_path_.empty() && !arena_map_dumped_
        && deps_.pinned_arena && deps_.numa_manager && deps_.live_config) {
        arena_map_dumped_ = true;
        const std::string tmp = arena_map_dump_path_ + ".tmp";
        if (FILE* f = std::fopen(tmp.c_str(), "w")) {
            const auto& mc = deps_.live_config->model;
            for (int g = 0; g < deps_.numa_manager->num_gpus(); ++g)
                std::fprintf(f, "g,%d,%d\n", g,
                             deps_.numa_manager->gpu_numa_node(g));
            auto paired = [&](int n) {
                if (n < 0 || !deps_.numa_manager->node_is_hbm(n)) return n;
                for (const auto& h : deps_.numa_manager->hbm_nodes())
                    if (h.node == n) return h.cpu_affinity_node;
                return n;
            };
            for (int l = mc.first_k_dense_replace; l < mc.num_hidden_layers;
                 ++l) {
                for (int e = 0; e < mc.n_routed_experts; ++e) {
                    const int n = deps_.pinned_arena->location_node(
                        make_key(static_cast<uint32_t>(l),
                                 static_cast<uint16_t>(e)));
                    // 4th field (additive, sscanf-compatible with the 3-field
                    // consumers): the RAW holding node — HBM nodes appear as
                    // themselves here, for placement-policy x-rays.
                    if (n >= 0)
                        std::fprintf(f, "e,%d,%d,%d,%d\n", l, e, paired(n), n);
                }
            }
            std::fclose(f);
            std::rename(tmp.c_str(), arena_map_dump_path_.c_str());
            spdlog::info("LS_ARENA_MAP_DUMP: wrote {} (gpu nodes + expert "
                         "host-slot nodes)", arena_map_dump_path_);
        }
    }
    // TASK-1 REEF path: when the engine's shadow solve is OFF (KEEPER52_REEF_ORCH
    // forces LS_LOADER_SHADOW=0 — the test orchestrates placement), the shadow
    // post-pass below never runs, so drive the B-aware never-lose CPU offload
    // directly off the REAL orchestrator placement here. Gated so exactly ONE
    // driver fires (shadow-on ⇒ the post-pass; shadow-off ⇒ this).
    if (loader_cpu_solver_ && !loader_shadow_) {
        const bool prof0 = mach_prof_.enabled;
        const uint64_t t0 = prof0 ? mach_prof_now() : 0;
        apply_cpu_offload_never_lose(entries, cmd.fetch_and_run_moe.expert_count,
                                     state, cmd.fetch_and_run_moe.layer_idx,
                                     cmd.fetch_and_run_moe.num_seqs);
        if (prof0) mach_prof_.add(LoaderMachProf::kResultOut, mach_prof_now() - t0);
    }
    if (!(loader_shadow_ && deps_.loader_constants &&
          deps_.loader_constants->num_devices > 0))
        return;
    // EVICTBOARD_EXTERNAL_SCORES: advance the daemon's per-token bookkeeping at the
    // EARLIEST loader-gated seam (token boundary = a non-increasing MoE layer
    // index, same rule feed_expert_stats uses). The EvictScoreBoard no longer
    // keeps an internal recency clock (scores are external); this only advances
    // stats_token_id_ for the off-hot-path ExpertStats feed. O(1).
    const bool new_token =
        advance_loader_token_if_new(cmd.fetch_and_run_moe.layer_idx);
    const auto& p = cmd.fetch_and_run_moe;

    // ALWAYS-ON per-access recency (TD-FAR-EVICT-REROUTE step 3). Advance the
    // board's recency clock once per layer-visit, then re-touch this layer's
    // ALREADY-resident routed experts (the cache hits) to the current clock so
    // they stay KEEP. Freshly-fetched misses are stamped at the SAME clock when
    // the cache reserves their slot (on_resident_added) — so the whole layer-visit
    // shares one recency value (per-layer-grouped LRU; within-layer ties ~K≈8 are
    // negligible vs ~460 slots). NO gate: the board now always carries a recency
    // signal, so score() conveys true LRU order even without LS_EVICT_DECAY.
    // touch_existing MUST NOT insert (a not-yet-resident miss would re-open desync
    // source (b)); the listener owns insertion.
    const bool prof = mach_prof_.enabled;
    uint64_t prof_t0 = prof ? mach_prof_now() : 0;
    if (evict_board_) {
        evict_board_->advance_recency();
        // Residency-aware touch (TD-LOADER-REUSE-ENGINE-FIDELITY): stamp EVERY
        // resident copy — under ACT a hit may execute on a replica away from the
        // orchestrator target, where the old target-keyed touch was a silent
        // no-op and the copy's recency went stale (evicted young; −4pp vs the
        // designed policy on the canonical trace). The stamp carries the
        // decayed-frequency bonus (policy lab): hot experts get up to ~8 tokens
        // of extra protection; the freq state decays per access, so it needs no
        // sweep. Misses update freq only (no resident copy to stamp yet — the
        // insert stamps plain clock, the next access adds the bonus).
        const double kFreqDecay = loader_freq_mult_;  // exp(-decay), hoisted
        for (uint32_t i = 0;
             i < p.expert_count && i < state.experts.size(); ++i) {
            const auto key = make_key(entries[i].layer_idx,
                                      entries[i].expert_idx);
            double bonus = 0.0;
            if (loader_freq_w_ > 0.0) {
                float& f = loader_freq_[key];
                f = static_cast<float>(f * kFreqDecay + 1.0);
                bonus = loader_freq_w_ * f;
            }
            evict_board_->touch_existing_all(key, bonus);
        }
        // EXPERIMENT (LS_EVICT_DECAY): on top of always-on recency, decay every
        // resident's raw once per token — a gentle uniform aging (order-preserving
        // per GPU) kept ONLY for A/B against the no-decay path.
        if (loader_evict_decay_ > 0.0 && new_token)
            evict_board_->decay_all(loader_evict_decay_);
    }
    if (prof) mach_prof_.add(LoaderMachProf::kTouch, mach_prof_now() - prof_t0);

#ifndef NDEBUG
    // Debug parity assert (TD-EVICT-BOARD-DESYNC): the board's per-GPU residency
    // must equal ExpertCache's stable residency. Trips immediately if any future
    // path mutates stable residency without the listener hearing.
    assert_board_cache_parity();
#endif

    std::vector<int>& solver_pos = loader_solver_pos_;  // persistent scratch
    shadow_solve_and_log(entries, p.expert_count, p.layer_idx, cmd.cmd_seq,
                         loader_act_ ? &solver_pos : nullptr, p.num_seqs);
    // I8 Stage-4: force-identity overrides the solver's j[·] with the orchestrator
    // e%tp target (entries[i].gpu_idx) AFTER the solve+plumbing ran, so the reroute
    // loop below is a no-op (sp == target_gpu) while every acting-path mechanism
    // (hot-path solve, FETCH-by-j bookkeeping, eviction plumbing) still executed.
    if (prof) prof_t0 = mach_prof_now();
    if (loader_force_identity_ && solver_pos.size() == state.experts.size())
        for (size_t i = 0; i < state.experts.size(); ++i)
            solver_pos[i] = static_cast<int>(entries[i].gpu_idx);
    if (loader_act_ && solver_pos.size() == state.experts.size()) {
        const int nb = static_cast<int>(deps_.device_backends.size());
        int rerouted = 0;
        for (size_t i = 0; i < state.experts.size(); ++i) {
            const int sp = solver_pos[i];
            if (sp >= 0 && sp < nb && !state.experts[i].was_cached &&
                sp != state.experts[i].target_gpu) {
                state.experts[i].target_gpu = sp;  // re-route this fetch (still a miss)
                ++rerouted;
            }
        }
        if (p.layer_idx == 0)
            spdlog::warn("LS_LOADER_ACT L{} rerouted {}/{} miss experts by solver",
                         p.layer_idx, rerouted, p.expert_count);
    }
    // TASK-2 never-lose bridge: shadow_solve_and_log has (re)populated
    // cpu_forced_experts_[layer] with THIS solve's CPU-column assignments (the
    // FRESH set — dispatch_moe's static is_cpu_forced check at state-build is
    // gated OFF in solver mode so route is the SOLE authority, no stale marking).
    // Mark those routed experts cpu_forced so the F-6 fetch decider skips their
    // GPU fetch and the finalize bitset excludes them; the fold path
    // (cpu_layer_has_forced / fold_cpu_forced_experts) computes them on host from
    // the same fresh set. At B=1 with the calibrated CPU cost this set is EMPTY
    // (zero beneficial offload) ⇒ NO fold ⇒ byte-identical to the no-CPU champion.
    if (loader_cpu_solver_) {
        auto it = cpu_forced_experts_.find(p.layer_idx);
        if (it != cpu_forced_experts_.end() && !it->second.empty()) {
            for (auto& er : state.experts) {
                if (er.cpu_forced || er.is_arrived) continue;  // only free misses
                const uint16_t eidx = er.key.expert_idx;
                for (uint16_t ce : it->second)
                    if (ce == eidx) {
                        er.cpu_forced = true;
                        ++state.cpu_forced_count;
                        break;
                    }
            }
        }
    }
    if (prof)
        mach_prof_.add(LoaderMachProf::kReroute, mach_prof_now() - prof_t0);
}

// Recency feed for the loader's evict_cum term. The token id advances per TOKEN,
// not per layer: a new token's MoE sequence restarts at the lowest MoE layer, so
// bump whenever layer_idx is not strictly greater than the previous one.
//
// LOADER_STATS_LOCALITY (resolves TD-EXPERTSTATS-FEED-COST): the daemon hot path
// NO LONGER round-trips through ExpertStats::update(). That feed cost ~18 ms/token
// (the 5.32→4.85 keeper regression — spec/REGRESSION_HUNT_5p2.md) NOT from its raw
// work but from scatter-touching the dense ~475 KB states_ array (and, on the
// solve side, fill_eviction_scores scanning ~460 residents/GPU/layer), evicting
// the daemon's hot working set. Instead we advance the EvictScoreBoard's OWN
// O(N)-contiguous token clock; the board derives recency/keep from its per-slot
// last-touch sequence (on_place stamps it), so the evict signal needs nothing
// from ExpertStats on the critical thread. LS_FEED_EXPERTSTATS still drives the
// full ExpertStats update for the off-hot-path / WorkloadDetector consumers and
// for A/B testing — but that is opt-in, never on the keeper's loader-on path.
void CommandDispatcher::feed_expert_stats(
    const ipc::Command& cmd, const ipc::ExpertPrefetchEntry* entries,
    bool have_weights) {
    static const bool force_feed = std::getenv("LS_FEED_EXPERTSTATS") != nullptr;
    if (!(loader_shadow_ || force_feed)) return;
    const auto& p = cmd.fetch_and_run_moe;

    // Hot-path recency: advance the board's own token clock once per token (O(1),
    // nothing dense). Idempotent with the earlier route_moe_by_loader call; this
    // covers the no-calibration path where route early-returned.
    advance_loader_token_if_new(p.layer_idx);

    // OFF-HOT-PATH (opt-in): the heavyweight ExpertStats dense feed, kept for the
    // WorkloadDetector / batched full-6-term consumers and A/B measurement only.
    if (!(force_feed && deps_.expert_stats)) return;
    statistics::GatingResult gr;
    gr.layer_idx = p.layer_idx;
    gr.token_id  = stats_token_id_;
    gr.activations.reserve(p.expert_count);
    for (uint32_t i = 0; i < p.expert_count; ++i) {
        const float w = (have_weights && i < p.weight_count) ? p.weights[i]
                                                             : 1.0f;
        gr.activations.push_back(statistics::ExpertActivation{
            make_key(entries[i].layer_idx, entries[i].expert_idx), w});
    }
    deps_.expert_stats->update(
        std::span<const statistics::GatingResult>(&gr, 1));
}

int CommandDispatcher::ensure_cpu_solver_k() {
    if (loader_cpu_k_built_) return loader_cpu_dev_pos_;
    loader_cpu_k_built_ = true;
    if (!deps_.loader_constants) return -1;
    loader_k_cpu_ = *deps_.loader_constants;  // deep copy of the live calibration
    gpu_loader::ComputeCurve cc;
    cc.a_us = loader_cpu_a_us_ * loader_cpu_cost_mult_;  // host FFN per-expert
    cc.b_us = loader_cpu_b_us_ * loader_cpu_cost_mult_;  // per-engaged-layer fold
    cc.P    = loader_cpu_p_;                             // fold charged once for c≤P
    loader_cpu_dev_pos_ =
        gpu_loader::append_cpu_expert_device(loader_k_cpu_, loader_cpu_solver_node_, cc);
    spdlog::warn("LS_LOADER_CPU_SOLVER: appended CPU device j={} pos={} to solver K "
                 "(now {} devices); ComputeCurve a={} b={} P={}",
                 loader_cpu_dev_pos_, loader_k_cpu_.devices[loader_cpu_dev_pos_].position,
                 loader_k_cpu_.num_devices, cc.a_us, cc.b_us, cc.P);
    return loader_cpu_dev_pos_;
}

namespace {
// ── TASK-1 B-aware never-lose CPU-offload greedy (shared by both drivers) ─────
// A routed miss (uncached expert i, home bank → its GPU jj, fetch = transfer+lat)
// is a candidate to compute on the host CPU device instead of the GPU. The
// decision is LAYER-LEVEL and B-aware (chunk width B): mirror the offline
// B-sweep sim's layer_wall_neverlose using the engine's calibrated K.
//   all-GPU wall = makespan over the M GPUs of ( Σ fetch of that GPU's misses +
//                  GPU compute of its assigned experts, batch-scaled ∝ B ).
//   offload k misses (greedily off the current bottleneck GPU, biggest fetch
//   first): the host FFN runs || the GPU window; its exposed part adds to the wall
//     wall(k) = gpu_wall'(k) + max(0, cpu_wall(k) − η·gpu_wall'(k))
//   cpu_wall(k) = fold + k·host_expert(B),  host_expert(B) = a_fixed + a_pertok·B.
//   η(B) = 1 − (1−floor)·exp(−(B−1)/τ).  pick k* = argmin_k wall(k); k=0 (all-GPU
//   champion) is ALWAYS in the search ⇒ NEVER worse (never-lose by construction).
// At B=1 (η≈floor, host FFN huge, no saturated channel to relieve) ⇒ k*=0.
//
// Residency reframe (loader_place_sum.h): each miss carries a `place_cost` = the
// CPU-column weighted-sum place_cons (residency baseline + numa + hotness). It is
// the HONEST future cost of offloading THIS expert (dominated by the extra cold
// fetch a would-recur HIT costs next round). The greedy (1) pops the biggest NET
// relief (fetch − place_cost) first — so residency-safe big-fetch experts offload
// before would-recur ones — and (2) ADDS the cumulative place_cost of the offloaded
// set to the wall. Since k=0 (place_cost 0) is always the incumbent, the priced
// term can only SHRINK the offload set ⇒ never-lose preserved, residency-aware.
struct CpuMiss { double fetch; double place_cost; uint32_t i; uint16_t eidx; };
struct CpuCurve { double a_fixed, a_pertok, fold, mult, eta_floor, eta_tau; };

// stacks[j] MUST be sorted ascending by fetch (pop from the back = biggest fetch).
// sub[j] = Σ fetch of GPU j's misses; cnt[j] = # experts computed on GPU j (all
// assigned, incl. hits — offloading a miss removes one). Fills `chosen` with the
// best_k greedy offload prefix; reports g0/best_wall/host_exp/fold/eta for logging.
int choose_cpu_offload(const gpu_loader::LoaderConstants& K, int M, uint32_t chunk_width,
                       const CpuCurve& cv,
                       std::array<std::vector<CpuMiss>, gpu_loader::kMaxDevices>& stacks,
                       double sub[], int cnt[], std::vector<CpuMiss>& chosen,
                       double& g0_out, double& wall_out, double& host_exp_out,
                       double& fold_out, double& eta_out) {
    const double Bd       = static_cast<double>(std::max<uint32_t>(1u, chunk_width));
    const double fold_us  = cv.fold * cv.mult;
    const double host_exp = (cv.a_fixed + cv.a_pertok * Bd) * cv.mult;
    const double eta = 1.0 - (1.0 - cv.eta_floor) * std::exp(-(Bd - 1.0) / cv.eta_tau);
    fold_out = fold_us; host_exp_out = host_exp; eta_out = eta;

    auto gpu_comp = [&](int j, int c) -> double {
        if (c <= 0) return 0.0;
        const auto& cc = K.devices[j].compute;
        return gpu_loader::compute_us(cc, c) + cc.a_us * static_cast<double>(c) * (Bd - 1.0);
    };
    auto makespan = [&]() -> double {
        double g = 0.0;
        for (int j = 0; j < M; ++j) g = std::max(g, sub[j] + gpu_comp(j, cnt[j]));
        return g;
    };
    int m_total = 0;
    for (int j = 0; j < M; ++j) {
        // Sort ascending by NET relief (fetch − residency place_cost): pop from the
        // back = biggest net first, so residency-SAFE big-fetch experts offload
        // before would-recur (high place_cost, near-zero net) ones.
        std::sort(stacks[j].begin(), stacks[j].end(),
                  [](const CpuMiss& a, const CpuMiss& b) {
                      return (a.fetch - a.place_cost) < (b.fetch - b.place_cost);
                  });
        m_total += static_cast<int>(stacks[j].size());
    }
    const double g0 = makespan();
    g0_out = g0; wall_out = g0;
    double best_wall = g0;
    int    best_k = 0;
    double cum_place = 0.0;  // Σ residency place_cost over the offloaded prefix
    std::vector<CpuMiss> order;
    order.reserve(static_cast<size_t>(m_total));
    std::array<size_t, gpu_loader::kMaxDevices> top{};
    for (int j = 0; j < M; ++j) top[j] = stacks[j].size();
    for (int k = 1; k <= m_total; ++k) {
        int jb = -1; double gb = -1.0;
        for (int j = 0; j < M; ++j) {
            if (top[j] == 0) continue;
            const double gj = sub[j] + gpu_comp(j, cnt[j]);
            if (gj > gb) { gb = gj; jb = j; }
        }
        if (jb < 0) break;
        const CpuMiss mm = stacks[jb][--top[jb]];
        sub[jb] -= mm.fetch;
        --cnt[jb];
        order.push_back(mm);
        cum_place += mm.place_cost;
        const double g        = makespan();
        const double cpu_wall = fold_us + static_cast<double>(k) * host_exp;
        const double exposed  = std::max(0.0, cpu_wall - eta * g);
        // Honest wall = GPU makespan + exposed host FFN + the future residency cost
        // of the offloaded set (would-recur HITs turned into next-round cold fetches).
        const double wall     = g + exposed + cum_place;
        if (wall < best_wall - 1e-9) { best_wall = wall; best_k = k; wall_out = wall; }
    }
    chosen.assign(order.begin(), order.begin() + best_k);
    return m_total;
}
}  // namespace

void CommandDispatcher::apply_cpu_offload_never_lose(
    const ipc::ExpertPrefetchEntry* entries, uint32_t n,
    ProgressiveMoeState& state, uint32_t layer_idx, uint32_t chunk_width) {
    if (!deps_.loader_constants) return;
    const auto& K = *deps_.loader_constants;
    // Adopt the trained place_cons weight from the calibration JSON (once). Only
    // w_numa is trained (residency+hotness are heuristics); it takes the trained
    // value unless LS_LOADER_PLACE_W_NUMA was set (env > trained-JSON > default).
    // Skipped when place_sum is explicitly OFF (LS_LOADER_PLACE_SUM=0 zeroed all).
    if (!loader_place_sum_calib_adopted_) {
        loader_place_sum_calib_adopted_ = true;
        if (loader_place_sum_ && K.place_sum_weights.present && !loader_place_w_numa_env_) {
            loader_place_sum_w_.w_numa = K.place_sum_weights.w_numa;
            spdlog::warn("LS_LOADER_PLACE_SUM: adopted TRAINED w_numa={} from "
                         "calibration JSON (residency+hotness are heuristics)",
                         loader_place_sum_w_.w_numa);
        }
    }
    const int M = K.num_devices;
    const int n_backends = static_cast<int>(deps_.device_backends.size());
    if (M != n_backends || n == 0 || M <= 0 || M > gpu_loader::kMaxDevices) return;
    cpu_forced_experts_[layer_idx].clear();  // fresh per solve (fold reads this exact set)
    if (chunk_width > loader_cpu_bmax_seen_) loader_cpu_bmax_seen_ = chunk_width;

    // C-6 Milestone C: B=1 (plain decode / draft) never offloads under never-lose
    // (host FFN ≫ single GPU fetch), so skip the solve entirely — the forced set
    // is already cleared (k=0). Removes the per-layer solver overhead from the
    // plain-step daemon critical path. LS_LOADER_CPU_SKIP_B1=0 forces the solve.
    if (loader_cpu_skip_b1_ && chunk_width <= 1) return;

    auto bank_of_node = [&](int node) {
        for (int b = 0; b < K.num_banks; ++b)
            if (K.banks[b].node == node) return b;
        return 0;
    };
    auto j_of_pos = [&](int pos) {
        for (int j = 0; j < M; ++j)
            if (K.devices[j].position == pos) return j;
        return -1;
    };

    // Residency reframe: the FREE b0_prev recurrence signal is this layer's
    // PREVIOUS-visit routed expert set. An expert in it is predicted to recur
    // within the ~1-token retention window (a would-be cache HIT ⇒ EXPENSIVE to
    // offload — offloading it excludes it from the GPU cache and pays a cold fetch
    // next round); absent ⇒ residency-safe (would-miss-anyway ⇒ free to offload).
    const bool place_sum_on = loader_place_sum_ && !loader_place_sum_w_.all_zero();
    loader_prev_round_set_.clear();
    if (place_sum_on) {
        auto pit = prev_round_experts_.find(layer_idx);
        if (pit != prev_round_experts_.end())
            for (uint16_t e : pit->second) loader_prev_round_set_.insert(e);
    }
    const int cpu_node = loader_cpu_solver_node_;

    // Build per-GPU miss stacks from the REAL orchestrator placement. cnt[j] = all
    // experts computed on GPU j (hits + misses); a miss = uncached (will be H2D-
    // fetched to its target) and not already CPU-forced. fetch = home-bank→GPU
    // transfer + latency (the wall it adds to that GPU's fetch channel). place_cost
    // = the CPU-column weighted-sum place_cons (residency baseline + numa + hotness).
    double sub[gpu_loader::kMaxDevices] = {0.0};
    int    cnt[gpu_loader::kMaxDevices] = {0};
    std::array<std::vector<CpuMiss>, gpu_loader::kMaxDevices> stacks;
    const size_t ne = std::min<size_t>(n, state.experts.size());
    std::vector<uint16_t> cur_round;  // this layer's routed set (next round's prev)
    if (place_sum_on) cur_round.reserve(ne);
    int tr_cand = 0;  // (e) target-node RARE candidates seen this solve (diagnostic)
    for (size_t i = 0; i < ne; ++i) {
        const auto& er = state.experts[i];
        const int j = j_of_pos(er.target_gpu);
        if (j < 0 || j >= M) continue;
        if (place_sum_on) cur_round.push_back(er.key.expert_idx);
        ++cnt[j];
        if (er.cpu_forced || er.was_cached || er.is_arrived) continue;  // hit / already off → keep
        const int node = deps_.numa_manager
            ? deps_.numa_manager->expert_home_node(er.key.expert_idx) : -1;
        // (e): when the target-node rare term is engaged, price the fetch from the
        // REAL arena holding node (location_node). A GPU-free spill node (node 1)
        // sources cross-node to EVERY GPU — the exact "most expensive to fetch"
        // premise the round-robin expert_home_node (nodes 0/2/3 only) cannot express.
        int loc_node = -1;
        if (loader_place_offload_node_ != -1 && deps_.pinned_arena)
            loc_node = deps_.pinned_arena->location_node(er.key);
        const int fetch_node = (loc_node >= 0) ? loc_node : node;
        const int b = (fetch_node >= 0) ? bank_of_node(fetch_node) : 0;
        const auto& cell = K.matrix[static_cast<size_t>(b)][j];
        const double fetch = cell.rate_us + cell.lat_us;
        // ── Weighted-sum CPU-column place_cons (loader_place_sum.h) ──────────
        double place_cost = 0.0;
        if (place_sum_on) {
            gpu_loader::PlaceSumFactors f;
            // (a) residency baseline: recur=1 if in the prev-round union.
            const double recur =
                loader_prev_round_set_.count(er.key.expert_idx) ? 1.0 : 0.0;
            f.resid_baseline = gpu_loader::residency_baseline(recur, fetch);
            // (b) NUMA home-locality: unit penalty (w_numa carries µs) if the
            // expert's home DDR node is not the CPU-FFN node (cross-node read).
            f.numa_penalty = (node >= 0 && node != cpu_node) ? 1.0 : 0.0;
            // (c) freq/EMA hotness: normalized M3 fetch frequency [0,1] if a freq
            // table is loaded (LS_LOADER_PLACE_HOTNESS_FREQ), else 0. Reused by (e).
            double h = 0.0;
            if (loader_place_hotness_use_freq_ && loader_place_hotness_fhot_ > 0.0) {
                auto fit = loader_place_hotness_freq_.find(er.key);
                const double fr = (fit != loader_place_hotness_freq_.end())
                                      ? static_cast<double>(fit->second) : 0.0;
                h = fr / loader_place_hotness_fhot_;
                if (h > 1.0) h = 1.0;
                f.hotness = h;
            }
            // (e) TARGET-NODE RARE: target_rare = 1 iff the expert's REAL holding
            // node (location_node) == the chosen GPU-free node AND it is globally
            // RARE (normalized M3 hotness h < rarity_thresh — INVERTED hotness). The
            // greedy then steers offloads to this residency-safe + host-light set
            // (offnode_protect PROTECTs the complement; encourage → 0 floor).
            if (loader_place_offload_node_ != -1 &&
                (loader_place_sum_w_.offnode_protect != 0.0 ||
                 loader_place_sum_w_.target_rare_encourage != 0.0) &&
                (loader_place_offload_node_ == -2 ||  // -2: ANY node (pure rarity)
                 loc_node == loader_place_offload_node_) &&
                h < loader_place_rarity_thresh_) {
                f.target_rare = 1.0;
                ++tr_cand;
            }
            // (d) NODE-BASED offload bias: home_hbm = 1 iff the expert's REAL arena
            // holding node (location_node — HBM banks report as themselves) is HBM.
            // This is the M3-hot proxy (globally hottest experts homed on HBM), NOT
            // the round-robin expert_home_node used by (b). Only queried when the
            // node term is engaged (both weights 0 ⇒ home_hbm irrelevant).
            if ((loader_place_sum_w_.hbm_offload_base != 0.0 ||
                 loader_place_sum_w_.ddr_offload_penalty != 0.0) &&
                deps_.pinned_arena && deps_.numa_manager) {
                const int hn = deps_.pinned_arena->location_node(er.key);
                if (hn >= 0 && deps_.numa_manager->node_is_hbm(hn)) f.home_hbm = 1.0;
            }
            place_cost = gpu_loader::cpu_place_cost(loader_place_sum_w_, f);
        }
        sub[j] += fetch;
        stacks[j].push_back({fetch, place_cost, static_cast<uint32_t>(i),
                             er.key.expert_idx});
    }

    const CpuCurve cv{loader_cpu_a_fixed_us_, loader_cpu_a_pertok_us_,
                      loader_cpu_b_us_, loader_cpu_cost_mult_,
                      loader_cpu_overlap_floor_, loader_cpu_overlap_tau_};
    std::vector<CpuMiss> chosen;
    double g0, best_wall, host_exp, fold_us, eta;
    const int m_total = choose_cpu_offload(K, M, chunk_width, cv, stacks, sub, cnt,
                                           chosen, g0, best_wall, host_exp, fold_us, eta);
    int cpu_this_layer = 0;
    int cpu_recur_off = 0;  // offloaded experts that were in the prev-round union
    int cpu_target_off = 0;  // (e) offloaded experts residing on the target node (LOCAL read)
    for (const auto& mm : chosen) {
        auto& er = state.experts[mm.i];
        if (er.cpu_forced) continue;
        er.cpu_forced = true;
        ++state.cpu_forced_count;
        cpu_forced_experts_[layer_idx].push_back(mm.eidx);
        ++cpu_this_layer;
        if (place_sum_on && loader_prev_round_set_.count(mm.eidx)) ++cpu_recur_off;
        // (e) LOCAL-READ audit: is this offloaded expert actually on the target node?
        if (loader_place_offload_node_ != -1 && deps_.pinned_arena) {
            // -2 (ANY node): "target-local" audit degrades to "on the CPU-FFN node"
            // (LOCAL host read); a specific node id keeps the exact-membership audit.
            const int want = (loader_place_offload_node_ == -2)
                                 ? cpu_node : loader_place_offload_node_;
            if (deps_.pinned_arena->location_node(er.key) == want) ++cpu_target_off;
        }
    }
    // Update the prev-round union for THIS layer (next round's b0_prev signal).
    if (place_sum_on) prev_round_experts_[layer_idx] = std::move(cur_round);
    loader_cpu_veto_total_ += static_cast<uint64_t>(m_total - cpu_this_layer);
    ++loader_cpu_solve_count_;
    loader_cpu_assign_total_ += static_cast<uint64_t>(cpu_this_layer);
    loader_cpu_recur_off_total_ += static_cast<uint64_t>(cpu_recur_off);
    loader_cpu_target_rare_cand_total_ += static_cast<uint64_t>(tr_cand);
    loader_cpu_target_off_total_ += static_cast<uint64_t>(cpu_target_off);
    if (cpu_this_layer > 0 || (loader_cpu_solve_count_ % 2000) == 0)
        spdlog::warn("LS_LOADER_CPU_SOLVER(REEF) L{} B={}: offloaded {}/{} miss(es) to "
                     "CPU ({} would-recur/HIT, {} on-target-node[{}]/{}-rare-cand) (wall "
                     "{:.0f}→{:.0f}us, host_exp={:.0f} fold={:.0f} η={:.2f}; cumulative {} "
                     "accepted ({} recur, {} target-local) / {} kept-on-GPU over {} solves; "
                     "target-rare-cands={}; Bmax={})",
                     layer_idx, chunk_width, cpu_this_layer, m_total, cpu_recur_off,
                     cpu_target_off, loader_place_offload_node_, tr_cand,
                     g0, best_wall, host_exp, fold_us, eta, loader_cpu_assign_total_,
                     loader_cpu_recur_off_total_, loader_cpu_target_off_total_,
                     loader_cpu_veto_total_, loader_cpu_solve_count_,
                     loader_cpu_target_rare_cand_total_, loader_cpu_bmax_seen_);
}

void CommandDispatcher::shadow_solve_and_log(
    const ipc::ExpertPrefetchEntry* entries, uint32_t n, uint32_t layer_idx,
    uint32_t cmd_seq, std::vector<int>* out_pos, uint32_t chunk_width) {
    if (out_pos) out_pos->assign(n, -1);  // default: no override (also on the skip paths below)
    // TASK-2: never-lose I8 CPU offload as a POST-PASS on the CHAMPION solve. The
    // GPU placement is solved EXACTLY as champion (no CPU column in K — appending
    // one lets the champion reuse VICTIM-PROTECTION reward, w=2000 µs, tip cold
    // misses to the place=0 CPU column and OVER-offload: measured a_us=1160 →
    // ~1.6 experts/layer, 5.3→2.7 tok/s, a LOSING cost-model artifact). Instead we
    // keep the champion GPU assignment byte-identical and, AFTER the solve, offload
    // a MISS to CPU ONLY when its CPU present transfer(0)+compute cost is strictly
    // below the GPU present transfer+compute it displaces — the never-lose
    // guarantee BY CONSTRUCTION (a losing offload is impossible; k=0 = champion).
    // ensure_cpu_solver_k() resolves the calibrated CPU ComputeCurve once. Clear
    // the dynamic per-layer CPU set up front so an all-hits fast-path return (no
    // misses ⇒ no offload) leaves NO stale forced expert for the fold.
    const int cpu_dev_pos = loader_cpu_solver_ ? ensure_cpu_solver_k() : -1;
    const bool cpu_solve = loader_cpu_solver_ && cpu_dev_pos >= 0;
    if (cpu_solve) cpu_forced_experts_[layer_idx].clear();
    // P-25: term ablation hands the solver zeroed cost inputs (identical struct
    // shape — device set / bank layout / positions untouched, so every guard and
    // index below is oblivious to the swap). mask==0 (default) = original K.
    const auto& K = loader_ablate_mask_ ? ablated_loader_constants()
                                        : *deps_.loader_constants;
    const int M = K.num_devices;
    const int B = K.num_banks;
    const int n_backends = static_cast<int>(deps_.device_backends.size());

    // Guard: solver assumes device index j == LoaderConstants.devices[j], so the
    // calibrated device set must match the live one. Skip (warn once) on mismatch
    // — regenerate the calibration JSON for the running GPU set, or run with all.
    if (M != n_backends || n == 0 || M > gpu_loader::kMaxDevices ||
        static_cast<int>(n) > gpu_loader::kMaxExperts) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            spdlog::warn("LS_LOADER_SHADOW: skipped — calib devices {} != live {} "
                         "(or N={} out of bounds); regenerate calibration for this GPU set",
                         M, n_backends, n);
        }
        return;
    }

    // NUMA node -> bank slot (banks[b].node); GPU position -> solver index j.
    auto bank_of_node = [&](int node) {
        for (int b = 0; b < B; ++b)
            if (K.banks[b].node == node) return b;
        return 0;  // unknown node -> bank 0 (shadow only)
    };
    auto j_of_pos = [&](int pos) {
        for (int j = 0; j < M; ++j)
            if (K.devices[j].position == pos) return j;
        return -1;
    };

    // LOADER_STATS_LOCALITY: build the top-layer evict/place boards on first use.
    ensure_loader_stats_boards();

    const bool prof = mach_prof_.enabled;
    uint64_t prof_t0 = prof ? mach_prof_now() : 0;
    gpu_loader::SolveRequest& req = loader_req_;  // persistent scratch (capacities survive)
    req.num_devices = M;
    req.num_experts = static_cast<int>(n);
    // clamp0 default (§5.4): place_cons -> max(0,·). The hotness lever with a
    // negative reward disables it (LS_LOADER_PLACE_HOTNESS_CLAMP=0); reset each
    // solve so a prior layer's override never leaks into the OFF/other paths.
    req.clamp_place =
        !(loader_place_hotness_ && !loader_place_hotness_clamp_);
    req.bank_of.resize(n);
    req.cached.assign(static_cast<size_t>(n) * M, 0);  // place/evict left 0 (shadow)
    // Pin cache HITS to the orchestrator's (resident) device: that is the
    // assignment ACT executes anyway (hits are never rerouted, cache-safe), and
    // it collapses the exact B&B to the miss subspace (TD-LOADER-SHADOW-HOTPATH-
    // COST: 4^8 -> 4^misses at the decode shape). An expert cached only AWAY
    // from its orchestrator target stays FREE — the solver may still reroute
    // that "miss" to the resident replica for a zero-cost fetch.
    if (loader_pin_hits_) req.pinned.assign(n, -1);
    else req.pinned.clear();
    // local→global flat expert ids for the N routed experts (place_cons gather).
    std::vector<int>& loader_globals = loader_globals_;
    loader_globals.assign(n, -1);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& e = entries[i];
        const int node = deps_.numa_manager
            ? deps_.numa_manager->expert_home_node(e.expert_idx) : -1;
        req.bank_of[i] = (node >= 0) ? bank_of_node(node) : 0;
        const auto key = make_key(e.layer_idx, e.expert_idx);
        loader_globals[i] = loader_flat_expert_id(e.layer_idx, e.expert_idx);
        for (int j = 0; j < M; ++j) {
            const int pos = K.devices[j].position;
            if (pos < 0 || pos >= n_backends) continue;  // defensive bounds
            const auto* ce = deps_.expert_cache->lookup(key, pos);
            if (ce && (ce->sub_components_ready & memory::SubComponent::kAll)
                        == memory::SubComponent::kAll)
                req.cached[static_cast<size_t>(i) * M + j] = 1;
        }
        if (loader_pin_hits_) {
            const int oj = j_of_pos(static_cast<int>(e.gpu_idx));
            if (oj >= 0 && req.cached[static_cast<size_t>(i) * M + oj])
                req.pinned[i] = oj;  // hit at the executed device -> fixed
            else if (loader_place_affinity_) {
                // P-25: the affinity policy rides residency WHEREVER the copy
                // lives — the replica is the synthetic objective's unique
                // 0-cost optimum for a cached-anywhere expert, so pinning it
                // is decision-neutral and collapses the B&B to the true-miss
                // subspace (without this, affinity placement makes most hits
                // live AWAY from their e%tp target → nfree≈8 → 271 µs solves,
                // measured 86 µs/layer vs REEF's ~19).
                for (int j = 0; j < M; ++j)
                    if (req.cached[static_cast<size_t>(i) * M + j]) {
                        req.pinned[i] = j;
                        break;
                    }
            }
        }
    }

    // ── M2v2 exposed-wall objective (LS_LOADER_M2) ──────────────────────────
    // Lazy one-shot load (needs M); enabled only with pinned hits — the solver
    // arms m2 on the pinned tier alone (bound validity, SolveRequest docs).
    // The persistent req keeps the param vectors across layers; only the bool
    // is (re)set per solve.
    if (!loader_m2_path_.empty() && !loader_m2_tried_) {
        loader_m2_tried_ = true;
        loader_m2_ = gpu_loader::load_m2_params(loader_m2_path_, M);
        if (loader_m2_.valid)
            spdlog::info("gpu_loader: M2v2 placement objective loaded ({})",
                         loader_m2_path_);
        else
            spdlog::warn("gpu_loader: LS_LOADER_M2 set but load failed ({}) — "
                         "legacy objective in use", loader_m2_path_);
    }
    req.m2 = loader_m2_.valid && loader_pin_hits_ && !loader_place_affinity_;
    if (req.m2 && req.m2_s.empty()) {
        req.m2_s = loader_m2_.s;
        req.m2_o0 = loader_m2_.o0;
        req.m2_oc = loader_m2_.oc;
        req.m2_hsat = loader_m2_.hsat;
        req.m2_cpw = loader_m2_.cpw;
        req.m2_gc = loader_m2_.g_c;
    }

    // ── All-hits fast path (perf-trace finding, 2026-07-18): a zero-miss layer
    // has nothing to decide — every expert is pinned to its resident target and
    // ACT reroutes nothing — yet the full path below (place fill, evict-curve
    // heap walks, solve) cost a measured ~211 µs/layer ≈ 3.4 ms/token on the
    // keeper (~23% of layers are all-hit at 0.74 hit rate). Skip it all; the
    // JSONL dump keeps the slow path so x-ray trajectories stay complete.
    if (loader_pin_hits_ && loader_shadow_dump_path_.empty() && !loader_shadow_log_) {
        bool all_pinned = true;
        for (uint32_t i = 0; i < n; ++i)
            if (req.pinned[i] < 0) { all_pinned = false; break; }
        if (all_pinned) {
            // Output the PINNED devices (== entries[i].gpu_idx for target-hit
            // pins; the resident replica for place-affinity cached-elsewhere
            // pins — the reroute loop then executes the hit at its replica).
            if (out_pos)
                for (uint32_t i = 0; i < n; ++i)
                    (*out_pos)[i] = K.devices[req.pinned[i]].position;
            // P-25 fine-tick mirror: an all-hit layer still re-touches every
            // resident in the keeper model (ticks advance) — stamp here too or
            // the fine ages go stale on skipped layers.
            if (loader_place_affinity_ && evict_board_) {
                std::array<int, gpu_loader::kMaxExperts> fp_pos;
                for (uint32_t i = 0; i < n; ++i)
                    fp_pos[i] = K.devices[req.pinned[i]].position;
                aff_fine_stamp(entries, n, fp_pos.data());
            }
            if (prof)
                mach_prof_.add(LoaderMachProf::kReqBuild,
                               mach_prof_now() - prof_t0);
            return;
        }
    }
    if (prof) {
        const uint64_t t = mach_prof_now();
        mach_prof_.add(LoaderMachProf::kReqBuild, t - prof_t0);
        prof_t0 = t;
    }

    // ── LOADER_STATS_LOCALITY: place_cons gather (B.2 — O(N) routed rows) ───
    // Read ONLY the N routed experts' rows from the contiguous [expert][gpu]
    // store (one cache line per row), in LOCAL index order — never the X-wide
    // table. The solver consumes req.place[i*M+j] via place_at(i,j), local i.
    // place_cons is currently a neutral-0 store (the cross-token reuse REWARD
    // policy is the separate open TD-LOADER-ROUTING-CROSSTOKEN), so this is a
    // no-op on the objective today — but it makes that future signal feedable on
    // the hot path with an O(N) footprint. clamp_place stays on (§5.4).
    if (place_table_) place_table_->gather(loader_globals, req.place);
    else req.place.clear();  // persistent scratch: never carry a stale layer's rows
    if (prof) {
        const uint64_t t = mach_prof_now();
        mach_prof_.add(LoaderMachProf::kPlaceGather, t - prof_t0);
        prof_t0 = t;
    }

    // ── place_cons cross-token reuse reward (TD-LOADER-ROUTING-CROSSTOKEN) ──
    // Per-device miss-placement cost pd[j] = w / (1 + age_j/τ), where age_j =
    // recency_now − cheapest_score(pos_j): the board's heap top is the pooled-LRU
    // victim (raw = last-touch layer clock), so age is BOUNDED and inflation-free
    // — unlike the evict term's raw magnitudes — and a trivialized duplicate's
    // ~0 effective score reads as an infinitely old (free) victim, the correct
    // semantics. Devices with stable free slots cost 0 (nothing displaced).
    // Charged per UNCACHED placement only (apply() skips place for hits), so
    // with pinned hits this steers exactly the miss subspace — the sim-validated
    // mechanism that recovers the affinity router's hit rate inside the solver.
    // P-25 (LS_LOADER_PLACE_AFFINITY): synthetic affinity-expressing objective —
    // replaces BOTH the reuse place reward and the evict-curve build below with
    // loader_affinity_place's count/age terms (see that header for the exact
    // greedy-router equivalence argument). Signals are frozen for the layer,
    // matching the router whose LRU state is constant while it assigns.
    gpu_loader::AffinityPlaceSignals aff_sig;
    if (loader_place_affinity_ && evict_board_) {
        aff_sig.num_devices = M;
        for (int j = 0; j < M; ++j) {
            const int pos = K.devices[j].position;
            if (pos < 0 || pos >= n_backends) continue;  // valid stays 0
            aff_sig.valid[j] = 1;
            const int fs = deps_.expert_cache
                ? deps_.expert_cache->free_slots(pos, memory::CacheZone::kStable)
                : 1;
            aff_sig.free_slot[j] = fs > 0;
            // Fine-tick age with the keeper router's needed_now exclusion:
            // this layer's hits on the device are un-evictable at decision time.
            std::array<memory::ExpertKey, gpu_loader::kMaxExperts> excl;
            int ne = 0;
            for (uint32_t i = 0; i < n; ++i)
                if (req.cached[static_cast<size_t>(i) * M + j])
                    excl[static_cast<size_t>(ne++)] =
                        make_key(entries[i].layer_idx, entries[i].expert_idx);
            aff_sig.victim_score[j] = aff_fine_oldest(pos, excl.data(), ne);
        }
        gpu_loader::build_affinity_place(aff_sig, req);
    } else if (loader_reuse_w_ > 0.0 && evict_board_) {
        const double now = evict_board_->recency_now();
        double pd[gpu_loader::kMaxDevices] = {0.0};
        bool any = false;
        for (int j = 0; j < M; ++j) {
            const int pos = K.devices[j].position;
            if (pos < 0 || pos >= n_backends) continue;
            const int fs = deps_.expert_cache
                ? deps_.expert_cache->free_slots(pos, memory::CacheZone::kStable)
                : 1;
            if (fs > 0) continue;                       // free slot → free placement
            double age = now - evict_board_->cheapest_score(pos);
            if (age < 0.0) age = 0.0;
            pd[j] = loader_reuse_w_ / (1.0 + age / loader_reuse_tau_);
            any = true;
        }
        if (any) {
            const size_t nm = static_cast<size_t>(n) * M;
            if (req.place.size() != nm) req.place.assign(nm, 0.0);
            for (uint32_t i = 0; i < n; ++i)
                for (int j = 0; j < M; ++j)
                    req.place[static_cast<size_t>(i) * M + j] += pd[j];
        }
    }

    // ── TASK-A2 phase 1: hotness-steered place_cons (LS_LOADER_PLACE_HOTNESS) ──
    // Adds w·coldness(i)·victim_hot(j) ON TOP of the gathered/reuse place term
    // (loader_place_hotness.h). coldness(i): 1 = the routed expert is resident on
    // NO device (a genuine cold-tail miss — the touch above already re-stamped
    // every resident copy to `now`, so a warm hit/replica reads coldness 0);
    // victim_hot(j) = reuse:inv of device j's cheapest-victim age (free slot ⇒ 0).
    // Mutually exclusive with the affinity synthetic objective (which owns place).
    if (loader_place_hotness_ && !loader_place_affinity_ && evict_board_) {
        const double now = evict_board_->recency_now();
        const double tau = loader_place_hotness_tau_;
        gpu_loader::HotnessPlaceSignals hsig;
        hsig.num_devices = M;
        hsig.w = loader_place_hotness_w_;
        for (int j = 0; j < M; ++j) {
            const int pos = K.devices[j].position;
            if (pos < 0 || pos >= n_backends) continue;  // valid stays 0
            hsig.valid[j] = 1;
            const int fs = deps_.expert_cache
                ? deps_.expert_cache->free_slots(pos, memory::CacheZone::kStable)
                : 1;
            if (fs > 0) { hsig.victim_hot[j] = 0.0; continue; }  // nothing displaced
            const double vage = now - evict_board_->cheapest_score(pos);
            hsig.victim_hot[j] = gpu_loader::reuse_inv_hot01(vage, tau);
        }
        std::vector<double>& cold = loader_hotness_cold_;  // persistent scratch
        cold.assign(n, 1.0);
        for (uint32_t i = 0; i < n; ++i) {
            const auto key = make_key(entries[i].layer_idx, entries[i].expert_idx);
            double coldness = 1.0;
            if (loader_place_hotness_use_freq_) {
                // TASK-1: FREQUENCY-PRIOR coldness. The M3 per-(layer,expert) fetch
                // count ranks EVERY routed expert (resident or not) — unlike board
                // eff which cannot score a cold-tail miss. Low freq ⇒ cold outlier
                // (1); high freq ⇒ hot (0). fhot = table max (or _FHOT override).
                auto it = loader_place_hotness_freq_.find(key);
                const double f = (it != loader_place_hotness_freq_.end())
                                     ? static_cast<double>(it->second) : 0.0;
                const double hot = (loader_place_hotness_fhot_ > 0.0)
                    ? f / loader_place_hotness_fhot_ : 0.0;
                coldness = 1.0 - hot;
                if (coldness < 0.0) coldness = 0.0;
                if (coldness > 1.0) coldness = 1.0;
            } else {
                double best_eff = gpu_loader::kAbsentScore;  // -1 ⇒ resident nowhere
                for (int j = 0; j < M; ++j) {
                    const int pos = K.devices[j].position;
                    if (pos < 0 || pos >= n_backends) continue;
                    const double s = evict_board_->score(pos, key);
                    if (s > best_eff) best_eff = s;
                }
                // Resident nowhere ⇒ maximally cold (1). Otherwise coldness =
                // 1 − reuse:inv(age of the warmest resident copy).
                if (best_eff >= 0.0) {
                    double age = now - best_eff;
                    coldness = 1.0 - gpu_loader::reuse_inv_hot01(age, tau);
                }
            }
            cold[i] = coldness;
        }
        gpu_loader::add_hotness_place(hsig, cold, req);
    }
    if (prof) {
        const uint64_t t = mach_prof_now();
        mach_prof_.add(LoaderMachProf::kReuse, t - prof_t0);
        prof_t0 = t;
    }

    // ── I8 P2: per-device convex eviction-cost term (evict_cum) ─────────────
    // Without this the solver is cache-capacity-blind (evict cost 0) and piles all
    // miss experts onto one GPU, evicting its hot residents. Build a CONVEX cost
    // curve per device from the live cache LRU/recency state so the solver prefers
    // to spread misses (cheap to evict a GPU's cold tail, expensive its hot set).
    //
    // Semantics (loader_solver objective_from_sums): evict_cum[j][n] is indexed by
    //   n = nunc_[j] = #uncached (newly-fetched) experts placed on device j
    // = #VRAM slots that must be freed. So free-slot headroom is folded INTO the
    // curve: cost 0 while n <= free_slots(j), then the prefix sum of the cheapest
    // victims' costs. Per-victim cost = unit * effective_score: a low-effective-
    // score resident (cheap to evict — an old/trivialized-duplicate copy) ≈ 0
    // cost, a high-effective-score resident (valuable, kept) ≈ unit. Victims
    // sorted ascending (cheapest first) ⇒ each prefix increment ≥ the previous ⇒
    // the curve is non-decreasing & convex, which makes the solver favour a
    // balanced 4/4 over an 8/0 overload on a bank-/makespan-tied instance.
    if (!loader_place_affinity_) {
        // unit µs: env LS_LOADER_EVICT_UNIT_US, else mean bank egress_us (a
        // representative cost to re-fetch a resident we just displaced). Falls back
        // to a fixed 600µs only if no bank egress is calibrated.
        double unit = loader_evict_unit_us_;
        if (unit <= 0.0) {
            double sum = 0.0; int cnt = 0;
            for (int b = 0; b < B; ++b) { sum += K.banks[b].egress_us; ++cnt; }
            unit = (cnt > 0 && sum > 0.0) ? (sum / cnt) : 600.0;
        }
        unit *= loader_evict_weight_;  // I8: horizon discount γ (future-token term)
        // γ==0 (the default since the reuse place term): every curve entry would
        // be unit·score·0 == 0 — identical to no evict term in the objective —
        // yet the build cost real per-layer heap walks (~1.3 µs/layer measured,
        // LS_LOADER_MACH_PROF) and an M-vector scan per B&B node. Hand the
        // solver an EMPTY evict_cum instead (its evict loop skips entirely).
        // LS_LOADER_EVICT_WEIGHT>0 restores the full build unchanged.
        if (unit == 0.0 || !evict_board_) {
            req.evict_cum.clear();
        } else {
        // Persistent inner vectors: resize the outer once, clear per-device below
        // (a skipped device must NOT keep a previous layer's curve).
        if (static_cast<int>(req.evict_cum.size()) != M) req.evict_cum.assign(M, {});
        const size_t curve_len = static_cast<size_t>(req.num_experts) + 1;
        // EVICTBOARD_EXTERNAL_SCORES (resolves TD-EXPERTSTATS-FEED-COST): the
        // per-victim effective scores come ENTIRELY from the top-layer
        // EvictScoreBoard's OWN O(N) state — NO eviction_inputs() resident-map
        // scan, NO ExpertStats fill_eviction_scores scattered states_ reads on the
        // daemon critical thread. The curve only needs the cheapest ≤ N victims
        // (the solver indexes vcost[evicts−1] with evicts ≤ N routed ≤
        // num_experts), which the board's effective-score min-heap surfaces in
        // O(N·log N) touching only ~N heap nodes. The effective score is
        // max(0, raw − rank·base): a low score (an old/trivialized-duplicate copy)
        // ⇒ cheapest victim, sign-preserving per INV-LOADER-OBJECTIVE-MYOPIC and
        // INV-0.4 (duplicates evict first). When no board (no calib / not built)
        // the device is left cache-capacity-blind (evict_cum empty), identical to
        // the pre-evict_cum behaviour.
        std::vector<double>& vcost = loader_vcost_;  // persistent scratch
        for (int j = 0; j < M; ++j) {
            const int pos = K.devices[j].position;
            if (pos < 0 || pos >= n_backends || !evict_board_) {
                req.evict_cum[static_cast<size_t>(j)].clear();  // stale-curve guard
                continue;
            }
            // Cheapest ≤ N residents' effective score, ascending (board keyed by
            // GPU POSITION, the same key the FETCH/evict triggers use).
            // resident_count drives the convex tail beyond the gathered victims.
            int resident_count = 0;
            const int need = req.num_experts;  // ≤ N evictions ever charged
            evict_board_->cheapest_scores_sorted(pos, need, vcost, &resident_count);
            for (double& v : vcost) v *= unit;          // score → µs cost
            const double max_v = vcost.empty() ? unit : vcost.back();
            const int fs = deps_.expert_cache
                ? deps_.expert_cache->free_slots(pos, memory::CacheZone::kStable)
                : resident_count;  // no cache → treat all residents as headroom
            auto& cum = req.evict_cum[j];
            cum.assign(curve_len, 0.0);
            double acc = 0.0;
            for (size_t n = 1; n < curve_len; ++n) {
                const int evicts = static_cast<int>(n) - fs;  // evictions needed at n arrivals
                if (evicts <= 0) { cum[n] = 0.0; continue; }
                // (evicts-1)-th cheapest victim; beyond the gathered victims, charge
                // the most-expensive seen per extra eviction (monotone/convex).
                const int vi = evicts - 1;
                const double step = (vi < static_cast<int>(vcost.size())) ? vcost[vi] : max_v;
                acc += step;
                cum[n] = acc;
            }
        }
        }  // else (unit != 0 && evict_board_)
    }

    if (prof) {
        const uint64_t t = mach_prof_now();
        mach_prof_.add(LoaderMachProf::kEvictCurve, t - prof_t0);
        prof_t0 = t;
    }
    auto r = loader_solver_.solve(K, req);
    if (prof) {
        const uint64_t t = mach_prof_now();
        mach_prof_.add(LoaderMachProf::kSolve, t - prof_t0);
        int nfree = 0;
        for (uint32_t i = 0; i < n; ++i)
            if (req.pinned.empty() || req.pinned[i] < 0) ++nfree;
        mach_prof_.add_solve(nfree, t - prof_t0);
        prof_t0 = t;
    }
    // P-25: normalize co-optimal miss pairings to the greedy router's
    // routed-order round-robin (objective-neutral — counts and the chosen
    // device multiset stay exactly the solver's; loader_affinity_place.h),
    // then stamp the fine-tick mirror with the FINAL executed positions.
    if (loader_place_affinity_ && evict_board_) {
        gpu_loader::canonicalize_affinity_pairing(aff_sig, req,
                                                  r.assignment.data());
        std::array<int, gpu_loader::kMaxExperts> st_pos;
        for (uint32_t i = 0; i < n; ++i) {
            const int jj = (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1;
            st_pos[i] = (jj >= 0 && jj < M) ? K.devices[jj].position : -1;
        }
        aff_fine_stamp(entries, n, st_pos.data());
    }

    // Always: solver j[.] -> GPU positions (the ACT output, CHAMPION-identical) +
    // per-device counts (the dump's dev_counts). Cheap integer work only.
    int sc[gpu_loader::kMaxDevices] = {0};
    for (uint32_t i = 0; i < n; ++i) {
        const int jj   = (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1;
        const int spos = (jj >= 0 && jj < M) ? K.devices[jj].position : -1;
        if (out_pos) (*out_pos)[i] = spos;
        if (jj >= 0 && jj < M) ++sc[jj];
    }
    // TASK-1/2 never-lose CPU offload POST-PASS, now B-AWARE (chunk_width). GPU
    // placement above is CHAMPION-identical; here we decide, at the LAYER level,
    // how many routed MISSES to offload to the host CPU device so the present
    // LAYER WALL drops. The decision mirrors the offline B-sweep sim's
    // layer_wall_neverlose exactly, using the engine's calibrated K:
    //
    //   all-GPU wall = makespan over the M GPUs of ( Σ fetch of that GPU's misses
    //                  + GPU compute of its assigned experts, batch-scaled ).
    //   offload k misses (greedily off the current bottleneck GPU, biggest fetch
    //   first): the host FFN runs || the GPU window; its exposed (unhidden) part
    //   adds to the wall —  wall(k) = gpu_wall'(k) + max(0, cpu_wall(k) − η·gpu_wall'(k)).
    //   cpu_wall(k) = fold(b_us) + k·host_expert(B),  host_expert(B) = a_fixed + a_pertok·B.
    //   η(B) = 1 − (1−floor)·exp(−(B−1)/τ)  — near-serial at B=1, saturates by B~16.
    //   pick k* = argmin_k wall(k);  k=0 (all-GPU champion) is ALWAYS in the search
    //   ⇒ NEVER worse (never-lose BY CONSTRUCTION).
    //
    // WHY B MATTERS: at plain decode (B=1) the union is small, the fetch channel
    // is not saturated (offloading relieves no fetch step), η≈floor and the host
    // FFN is huge ⇒ k*=0 (ZERO offload — byte-identical champion, matching the
    // Task-2 engine boot). At the dsp52 batched verify chunk (B=1+γ=16) the union
    // is ~4× larger and STACKS ~8 misses on the busiest GPU: each 4 offloaded
    // removes a whole ~fetch step from the saturated channel while the host FFN
    // (per-token cost collapsed to ~71 µs, fixed weight-read/coord amortized over
    // the chunk) HIDES under the ~B× GPU window ⇒ the wall drops and the champion
    // wins. LS_LOADER_CPU_COST_MULT<1 scales the host FFN down (mechanism-live probe).
    int cpu_this_layer = 0;
    if (cpu_solve) {
        (void)cpu_dev_pos;  // appended CPU device kept for the mechanism-live probe/display
        if (chunk_width > loader_cpu_bmax_seen_) loader_cpu_bmax_seen_ = chunk_width;
        // Build the per-GPU miss stacks from the CHAMPION shadow assignment r.
        double sub[gpu_loader::kMaxDevices] = {0.0};
        int    cnt[gpu_loader::kMaxDevices];
        for (int j = 0; j < M; ++j) cnt[j] = sc[j];
        std::array<std::vector<CpuMiss>, gpu_loader::kMaxDevices> stacks;
        for (uint32_t i = 0; i < n; ++i) {
            if (!req.pinned.empty() && req.pinned[i] >= 0) continue;   // hit → fixed on GPU
            const int jj = (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1;
            if (jj < 0 || jj >= M) continue;
            if (req.cached[static_cast<size_t>(i) * M + jj]) continue; // resident on jj (0-fetch)
            const auto& cell = K.matrix[static_cast<size_t>(req.bank_of[i])][jj];
            const double fetch = cell.rate_us + cell.lat_us;
            sub[jj] += fetch;
            stacks[jj].push_back({fetch, i, entries[i].expert_idx});
        }
        const CpuCurve cv{loader_cpu_a_fixed_us_, loader_cpu_a_pertok_us_,
                          loader_cpu_b_us_, loader_cpu_cost_mult_,
                          loader_cpu_overlap_floor_, loader_cpu_overlap_tau_};
        std::vector<CpuMiss> chosen;
        double g0, best_wall, host_exp, fold_us, eta;
        const int m_total = choose_cpu_offload(K, M, chunk_width, cv, stacks, sub, cnt,
                                               chosen, g0, best_wall, host_exp, fold_us, eta);
        for (const auto& mm : chosen) {
            cpu_forced_experts_[layer_idx].push_back(mm.eidx);
            if (out_pos) (*out_pos)[mm.i] = -1;   // fold owns it
            ++cpu_this_layer;
        }
        loader_cpu_veto_total_ += static_cast<uint64_t>(m_total - cpu_this_layer);
        ++loader_cpu_solve_count_;
        loader_cpu_assign_total_ += static_cast<uint64_t>(cpu_this_layer);
        if (cpu_this_layer > 0 || (loader_cpu_solve_count_ % 2000) == 0)
            spdlog::warn("LS_LOADER_CPU_SOLVER L{} B={}: offloaded {}/{} miss(es) to CPU "
                         "(wall {:.0f}→{:.0f}us, host_exp={:.0f} fold={:.0f} η={:.2f}; "
                         "cumulative {} accepted / {} kept-on-GPU over {} solves; Bmax={})",
                         layer_idx, chunk_width, cpu_this_layer, m_total, g0, best_wall,
                         host_exp, fold_us, eta, loader_cpu_assign_total_,
                         loader_cpu_veto_total_, loader_cpu_solve_count_,
                         loader_cpu_bmax_seen_);
    }
    if (prof)
        mach_prof_.add(LoaderMachProf::kResultOut, mach_prof_now() - prof_t0);
    // Compact per-layer log (solver vs orchestrator positions + predicted
    // breakdown): DIAGNOSTIC ONLY, and the per-expert string building + sink
    // write cost real hot-path time at 75 layers/token — gated behind
    // LS_LOADER_SHADOW_LOG (default off; TD-LOADER-SHADOW-HOTPATH-COST).
    if (loader_shadow_log_) {
        int oc[gpu_loader::kMaxDevices] = {0};
        std::string sj, oj;
        for (uint32_t i = 0; i < n; ++i) {
            const int jj   = (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1;
            const int spos = (jj >= 0 && jj < M) ? K.devices[jj].position : -1;
            const int opos = static_cast<int>(entries[i].gpu_idx);
            const int ojx = j_of_pos(opos);
            if (ojx >= 0) ++oc[ojx];
            sj += std::to_string(spos); sj += ' ';
            oj += std::to_string(opos); oj += ' ';
        }
        std::string scnt, ocnt;
        for (int j = 0; j < M; ++j) {
            scnt += std::to_string(sc[j]); scnt += ' ';
            ocnt += std::to_string(oc[j]); ocnt += ' ';
        }
        spdlog::warn("LS_LOADER_SHADOW L{} N={} M={}  solver_pos=[{}] orch_pos=[{}]  "
                     "counts solver=[{}] orch=[{}]  T={:.1f}us(dev={:.1f} bank={:.1f}) exact={}",
                     layer_idx, n, M, sj, oj, scnt, ocnt,
                     r.predicted_us, r.device_makespan_us, r.bank_egress_us, r.exact);
    }

    // ── I8b model-vs-reality x-ray: structured JSONL dump (env LS_LOADER_SHADOW_DUMP) ──
    // One record per solve, keyed by (cmd_seq, layer_idx) — the join key against the
    // perf_trace CSV. Captures the model's full INPUT (per-expert bank / cache mask /
    // subxfer[j] / egress) → OUTPUT (j[·], all sub-terms, per-device counts, exact).
    // Zero cost when the env is unset (path empty). Hand-rolled JSON (no JSON lib on
    // the daemon hot path; matches house manual-builder style). CUDA-free.
    if (loader_shadow_dump_path_.empty()) return;
    if (!loader_shadow_dump_fp_) {
        loader_shadow_dump_fp_ = std::fopen(loader_shadow_dump_path_.c_str(), "w");
        if (!loader_shadow_dump_fp_) {
            spdlog::warn("LS_LOADER_SHADOW_DUMP: failed to open {} — dump disabled",
                         loader_shadow_dump_path_);
            loader_shadow_dump_path_.clear();
            return;
        }
    }
    FILE* f = loader_shadow_dump_fp_;
    std::fprintf(f, "{\"cmd_seq\":%u,\"layer_idx\":%u,\"N\":%u,\"M\":%d,\"B\":%d,",
                 cmd_seq, layer_idx, n, M, B);
    // per-expert array: bank, cached mask (per device), subxfer[j] (dest ingest), egress (source-bank draw), assigned j
    std::fprintf(f, "\"experts\":[");
    for (uint32_t i = 0; i < n; ++i) {
        const int bank = req.bank_of[i];
        const int jj   = (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1;
        // egress draw for this expert: the RAW per-bank egress rate, ASSIGNMENT-
        // INDEPENDENT (the source bank is fixed per expert). The offline x-ray zeroes
        // it per-candidate when the expert is cached on the assigned device. (Earlier
        // dumps zeroed it for the solver's j here; emitting the raw rate lets Stage-1
        // recompute egress for ANY assignment — incl. the executed e%tp — offline.)
        const double eg = (bank >= 0 && bank < B) ? K.banks[bank].egress_us : 0.0;
        // I8 Stage-1: record the EXECUTED (orchestrator e%tp) device index alongside
        // the solver's hypothetical j, so the offline Σ-calibration can recompute
        // predicted-T for the ASSIGNMENT THAT RAN (which under ACT off ≠ solver's j).
        // oj = solver device index of the orchestrator's gpu_idx (-1 if unmapped).
        const int oj = j_of_pos(static_cast<int>(entries[i].gpu_idx));
        std::fprintf(f,
                     "%s{\"bank\":%d,\"j\":%d,\"oj\":%d,\"expert_idx\":%u,"
                     "\"egress_us\":%.4f,\"cached\":[",
                     i ? "," : "", bank, jj, oj,
                     static_cast<unsigned>(entries[i].expert_idx), eg);
        for (int j = 0; j < M; ++j)
            std::fprintf(f, "%s%d", j ? "," : "",
                         req.cached[static_cast<size_t>(i) * M + j] ? 1 : 0);
        std::fprintf(f, "],\"subxfer_us\":[");
        for (int j = 0; j < M; ++j) {
            // mirror solver subxfer_us: 0 if cached on j, else matrix[bank][j].rate+lat
            double sx = 0.0;
            if (!req.cached[static_cast<size_t>(i) * M + j] && bank >= 0 && bank < B) {
                const auto& cell = K.matrix[bank][j];
                sx = cell.rate_us + cell.lat_us;
            }
            std::fprintf(f, "%s%.4f", j ? "," : "", sx);
        }
        std::fprintf(f, "]}");
    }
    std::fprintf(f, "],");
    // assignment j[.] (solver device index), per-device counts, predicted breakdown.
    std::fprintf(f, "\"j\":[");
    for (uint32_t i = 0; i < n; ++i)
        std::fprintf(f, "%s%d", i ? "," : "",
                     (i < static_cast<uint32_t>(r.n)) ? r.assignment[i] : -1);
    std::fprintf(f, "],\"dev_counts\":[");
    for (int j = 0; j < M; ++j) std::fprintf(f, "%s%d", j ? "," : "", sc[j]);
    std::fprintf(f, "],");
    std::fprintf(f,
                 "\"predicted_us\":%.4f,\"device_makespan_us\":%.4f,"
                 "\"bank_egress_us\":%.4f,\"recon_us\":%.4f,\"place_us\":%.4f,"
                 "\"evict_us\":%.4f,\"prep_us\":%.4f,\"exact\":%s}\n",
                 r.predicted_us, r.device_makespan_us, r.bank_egress_us, r.recon_us,
                 r.place_us, r.evict_us, r.prep_us, r.exact ? "true" : "false");
    std::fflush(f);
}

}  // namespace layerstorm::daemon
