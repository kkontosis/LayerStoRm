#pragma once

// ── 44z stage 4: KV <-> expert zone REBALANCER ─────────────────────────────
// (spec/tickets/44z_KV_EXPERT_ZONE_REBALANCE.md §2/§4/§5)
//
// The policy layer above the two mechanisms stages 2 and 3 landed:
// PageAllocator::claim_expert_zone/release_expert_zone (a contiguous run of
// whole kMain slabs lent out of the shared region) and
// ExpertCache::add_elastic_zone/remove_elastic_zone (that run presented to
// the cache as extra kStable slots). This class decides WHEN a run moves.
//
// ── THE ASYMMETRY ─────────────────────────────────────────────────────────
// An expert slot is a CACHE LINE over host RAM: reclaiming it costs a
// re-fetch (~0.46 ms) and destroys nothing. A KV page is STATE: no host copy
// exists to re-read, so dropping one destroys work already paid for.
// Pressure therefore flows expert -> KV and NEVER the reverse. Every rule
// below is a consequence of that one fact.
//
// ── HYSTERESIS (user-required) ────────────────────────────────────────────
// A reclaim must NEVER be on the critical path. Concretely:
//   * a LOW-WATER mark on free KV slabs triggers a BACKGROUND reclaim before
//     anybody blocks — the drain runs across daemon ticks while KV still has
//     slabs to hand out;
//   * grants happen only ABOVE the HIGH-WATER mark, so a grant can never
//     itself push the pool under low water and immediately trigger the
//     reclaim it just caused;
//   * the band between the two marks is what prevents oscillation (grant ->
//     reclaim -> grant) around a single threshold.
// Grants are periodic + opportunistic (only when the pool is comfortable);
// reclaims are eager + anticipatory (started at the first sign of pressure).
//
// ── BAND-WIDTH GUARD (measured, not theoretical) ──────────────────────────
// The band only works if it is WIDER than the largest normal excursion of
// free slabs. One admission transiently claims a whole KDA-state footprint
// (mapped: units_per_rank * unit_slabs) plus its indexer share before the
// sequence settles. With marks set 3 percentage points apart the band came
// out NARROWER than that single transient, so every admission traversed the
// entire band: claim -> free below low -> reclaim two zones -> sequence
// settles -> free above high -> re-grant the SAME span -> next admission
// repeats. Measured cost of that arm: 8 residents evicted mid-service per
// cycle, 12 layers degraded past the CPU-overlap deadline, decode 1.54 tok/s
// (vs a rock-stable default band 2.4x the transient).
// So the marks are DERIVED and CHECKED, never left to configuration luck.
//
// ── The marks are ABSOLUTE, not a fraction of the pool ────────────────────
// What must stay free is the in-flight admission transient plus the band.
// That is a property of ONE ADMISSION, not of how big the pool is. A
// fraction-shaped reserve strands proportionally MORE capacity the emptier
// the pool is — exactly backwards, since an idle pool is precisely when the
// expert cache should be allowed to borrow. On the measured arm a 0.30
// fraction held 4232 slabs (~1.15 GB, ~62 expert slots) hostage where the
// absolute derivation needs 2312.
//
// ── THE PER-STEP FLOOR (TD-KVXP-PER-STEP-FLOOR, user design) ─────────────
// KV grows INCREMENTALLY: a decode step claims ~one page per KV-bearing
// layer when it crosses a page boundary (~0.75 pages/token on glm5_next =
// one slab per ~44 steps per sequence), so the standing reserve does not
// need to hold a whole admission's KV — only the NON-DEFERRABLE upfront
// part (the mapped KDA state, claimed whole at seq_create) plus per-step
// growth headroom for the sequences already running:
//
//   low  = max(max_concurrent_admissions
//                  * (admission_transient_slabs + per_step_growth_slabs),
//              ceil(low_frac  * total))
//   band = 2 * per_step_growth_slabs
//   high = max(low + band, ceil(high_frac * total))
//
// The one case the incremental argument breaks — a BULK PREFILL's upfront
// KV/indexer claim — is deliberately NOT covered by a permanent reserve:
// above a large-prefill threshold the orchestrator answers the retryable
// refusal with a BOUNDED WAIT on the eager drain this class arms (see
// note_pool_pressure_refusal), which costs ~one drain (~400 ms measured)
// against a >= 1 s prefill. Below the threshold, and on every decode step,
// behavior is exactly the pre-floor seam (evict-retry, never a wait). The
// wait lives ABOVE the admission protocol — every engine-side claim stays
// all-or-nothing with full rollback (INV-KDA-STATE (a) untouched).
//
// When per_step_growth_slabs is UNWIRED (0), the legacy derivation applies:
//   low  = max(max_concurrent_admissions * admission_transient_slabs,
//              ceil(low_frac  * total))
//   band = max(2 * admission_transient_slabs,
//              2 * grant_slabs(max_slots_per_grant, total))
//
// The low mark covers the engine's OWN ADMISSION PARALLELISM: the reserve
// exists so that max_concurrent_requests simultaneous admissions' upfront
// state claims (plus growth headroom) can all land without waiting on a
// reclaim — the same worst-case multiplier the KV pool itself is sized by,
// so the two agree by construction instead of by a hardcoded constant.
// The per-step band is deliberately NARROWER than one admission's state
// excursion (that is the whole point — the legacy band held 2 x 578 slabs
// hostage on the measured arm, the per-step band holds 2 x 36): an
// admission landing near the marks may now start a background drain cycle,
// and that is accepted by design — drains are background, the grant
// cooldown (30 s) bounds re-grant churn, and the eager drain + bounded
// wait backstop the rare bulk claim. Measured arm delta: high 2,312 ->
// 1,300 slabs, ~15 expert slots recovered.
// The fractions survive ONLY as optional floors a user can raise — defaults
// 0.02 / 0.0 mean the marks come out of the transient alone. A requested high floor narrower than one transient
// above low water is ignored with a warn; if the marks cannot fit the pool
// at all, the rebalancer disables itself for the boot.
//
// A post-reclaim GRANT COOLDOWN (grant_cooldown_ms) damps the residual
// band-edge thrash. It gates grants only — a reclaim is never delayed.
//
// A genuinely FORCED immediate reclaim — an admission refused with the
// retryable kKvPoolExhausted class while grants are still outstanding — is a
// POLICY FAILURE for SMALL claims: the low-water mark was set too low to
// start the background drain in time. It is COUNTED
// (Stats::forced_immediate_reclaims) precisely so the mis-set mark is
// visible rather than silently absorbed. See note_pool_pressure_refusal().
// TD-KVXP-PER-STEP-FLOOR nuance: a LARGE prefill's bulk upfront claim rides
// this same seam BY DESIGN (the per-step floor deliberately does not
// reserve for it; the orchestrator's bounded wait consumes the eager drain
// this call arms), so under the per-step marks the counter reads "refusals
// answered by a drain", and only refusals on small claims — visible in the
// seq_create WARN lines' prompt_len — indicate a mis-set mark.
//
// ── THE RECLAIM PROTOCOL (the hazard) ─────────────────────────────────────
// Returning slabs to KV while any dispatch can still touch the region would
// corrupt KV state, so the handback is a five-step ratchet in which only the
// LAST step returns capacity; every earlier step is reversible bookkeeping:
//   (1) begin_drain_elastic_zone — the zone refuses NEW reserves. Existing
//       residents stay valid and keep serving.
//   (2) evict residents as they become ready + unlocked, THROUGH the
//       lifecycle manager (Deps::evict_expert -> ELM::request_evict, never
//       ExpertCache::evict — INV-ELM-EVICT). A locked entry (#90, an
//       in-flight MoE dispatch holds it) or an ELM refusal simply defers the
//       drain to a later tick — never a forced eviction.
//   (3) once the zone is drained, wait for MoE DISPATCH QUIESCE — meaning NO
//       MoE KERNEL IS IN FLIGHT, deliberately NOT "the pipeline is idle".
//       The idle reading was tried and DEADLOCKED (measured: engine wedged
//       9+ min, GPU 0%, drain stuck at START): a prefill stalled on a KV page
//       claim holds its progressive state ACTIVE while waiting — for the very
//       slabs this drain would hand back. Drain waited on quiesce, quiesce on
//       the prefill, the prefill on the drain. The weaker predicate suffices
//       because the daemon is single-threaded (this tick runs only AFTER a
//       dispatch call returns, never between a pointer-table fill and its
//       launch) and every cross-call surface is refilled per use — see
//       CommandDispatcher::moe_dispatch_quiesced() for the full argument.
//       An active-but-stalled state is PRE-DISPATCH: its next fill re-reads
//       the cache and sees evicted entries as ordinary misses.
//   (4) record barrier events on the h2d transfer stream AND on the
//       expert-FFN compute stream, per rank. TransferEngine::cancel does NOT
//       stop a DMA already dispatched to the copy engine — only a stream
//       barrier proves the writes INTO the region have landed.
//   (5) when those events complete: remove_elastic_zone, then
//       release_expert_zone. Now, and only now, the slabs are KV's again.
//
// ── TP LOCKSTEP ───────────────────────────────────────────────────────────
// Grants are claimed BY SLAB ID on every attention-host rank (rank 0 chooses,
// the others mirror with claim_expert_zone_at). Replicated KV (glm5_next)
// claims pages in lockstep by index, so per-rank-divergent free sets collapse
// replicated capacity to the INTERSECTION of the ranks' free slabs — the S4 /
// INV-KVT-14b lesson. Any mirror failure rolls the WHOLE grant back.
//
// Default OFF (LS_KV_EXPERT_REBALANCE=1 to enable). Daemon-thread only, not
// thread-safe: ticked from the DaemonLoop background hook and poked from the
// dispatcher's admission seam, both on the daemon thread (INV-3.4.2).

#include <cstdint>
#include <chrono>
#include <functional>
#include <vector>

#include "core/memory/eviction_policy.h"  // memory::ExpertKey (evict_expert dep)
#include "core/memory/expert_zone_math.h"

namespace layerstorm::memory {
class PageAllocator;
class ExpertCache;
}  // namespace layerstorm::memory

namespace layerstorm::daemon {

/// Collaborators + the injectable device surface. The stream/event functions
/// are injected (rather than taking TransferEngine/DeviceBackend directly) so
/// the unit suite can drive the whole protocol CPU-only: the engine wires the
/// real TransferEngine + per-GPU DeviceBackend, tests wire counting stubs.
struct ExpertZoneRebalancerDeps {
    memory::PageAllocator* page_allocator = nullptr;
    memory::ExpertCache* expert_cache = nullptr;
    /// Attention-host GPUs, rank 0 first (the rank that CHOOSES the run).
    std::vector<int> tp_gpus;
    /// Model-derived slot/slab geometry; max_waste comes from the env config.
    memory::ExpertZoneGeometry geometry;
    /// True iff no MoE KERNEL is in flight
    /// (CommandDispatcher::moe_dispatch_quiesced). Step (3) of the protocol.
    /// NOT "the pipeline is idle" — see the protocol note above: demanding
    /// idleness deadlocks a memory-stalled admission against its own drain.
    std::function<bool()> moe_quiesced;
    /// Record a barrier event at the current tail of the GPU's h2d stream.
    /// Returns the event (caller destroys it), or nullptr on failure.
    std::function<void*(int)> record_h2d_barrier;
    /// Record a barrier event on the GPU's kExpertFfn compute stream.
    std::function<void*(int)> record_ffn_barrier;
    /// Non-blocking completion query for an event on a GPU.
    std::function<bool(int, void*)> event_complete;
    /// Destroy an event previously produced by the two record functions.
    std::function<void(int, void*)> destroy_event;
    /// Evict one READY + UNLOCKED zone resident during step (2) of the drain.
    /// The engine wires ExpertLifecycleManager::request_evict — NEVER
    /// ExpertCache::evict directly (INV-ELM-EVICT): the ELM tracks a
    /// per-(key,gpu) lifecycle tier ABOVE the cache, and an eviction the ELM
    /// does not see leaves that tier kHot for a key the cache no longer
    /// holds. ensure_resident then short-circuits ("already resident", no
    /// fetch is ever started), the progressive arrival poll — which reads
    /// the CACHE — never sees an arrival, and every MoE layer routing the
    /// key silently burns its full fetch deadline: the
    /// TD-KVXP-RECLAIM-REGRANT-WEDGE crawl (GPU 0%, zero log lines, looks
    /// like a hard hang). request_evict may REFUSE (pending interests,
    /// kTransferring/kDraining) — the drain then simply retries on a later
    /// tick, the same deferral the READY+UNLOCKED gate already produces.
    /// Unset => direct ExpertCache::evict (ELM-less unit suites only).
    std::function<bool(memory::ExpertKey, int)> evict_expert;
    /// Slabs ONE admission transiently claims before the sequence settles
    /// (mapped KDA state runs; the engine wires it). The band-width guard
    /// below needs it: a hysteresis band NARROWER than one admission's
    /// transient footprint is not hysteresis at all — every single admission
    /// crosses the whole band, so the rebalancer reclaims on the way down and
    /// re-grants the same span on the way up, once per request. 0 = unknown
    /// (the guard then falls back to the two-grants-wide floor alone).
    int64_t admission_transient_slabs = 0;
    /// TD-KVXP-PER-STEP-FLOOR: slabs ONE incremental growth event claims —
    /// one KV auto-growth chunk (page_growth_chunk_tokens worth of logical
    /// pages x the kMain-KV-bearing layer count, ceiled to slabs) plus one
    /// indexer-K growth group (one slab per indexer-computing layer on
    /// paged-indexer models). The engine wires
    /// CommandDispatcher::growth_chunk_slabs(). When > 0 the marks use the
    /// per-step derivation (see the header): the low mark adds this per
    /// concurrent admission as growth headroom for RUNNING sequences, and
    /// the band shrinks to 2x this value — the bulk-prefill claim the old
    /// band covered is answered by the orchestrator's bounded large-prefill
    /// wait instead. 0 = unwired (legacy transient-derived marks).
    int64_t per_step_growth_slabs = 0;
    /// How many admissions the engine may have in flight at once
    /// (serving.max_concurrent_requests — the SAME worst-case multiplier the
    /// KV pool itself is sized by). The low-water reserve is
    /// max_concurrent_admissions * (admission_transient_slabs +
    /// per_step_growth_slabs): the reserve exists so that this many
    /// simultaneous admissions' non-deferrable upfront state claims — plus
    /// growth headroom for the sequences already running — can ALL land
    /// without waiting on a reclaim. At B=1-class configs (2) the state
    /// term equals the 2x that was previously hardcoded, so measured arms
    /// stay valid.
    int max_concurrent_admissions = 2;
};

/// Policy knobs: schema section _internal-kv_expert_rebalance (defaults
/// below mirror it), with the LS_* env vars overriding either way
/// (from_config_env). Everything is OFF by default.
struct ExpertZoneRebalancerConfig {
    /// 44z master switch. DEFAULT ON since 2026-09-03 (user decision on the
    /// glm5_next EP4 decisive A/B: ON was the TIGHTEST of six arms, recovered
    /// 99.1% of a withheld carve at matched fetch volume, text byte-identical
    /// across 66 runs). The value is ELASTICITY; the wall gain is modest and
    /// shape-dependent. Known wart: TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION.
    bool enabled = true;            ///< LS_KV_EXPERT_REBALANCE=0 disables
    double max_waste = 0.10;        ///< LS_KVXP_MAX_WASTE (44z §3 bound)
    /// LS_KVXP_LOW_FRAC / LS_KVXP_HIGH_FRAC — optional pool-fraction FLOORS
    /// under the absolute marks, NOT the marks themselves (see the mark
    /// derivation in the class comment). Default low 0.02 / high 0.0 means
    /// "no meaningful floor": the marks come out of the admission transient
    /// alone. Raising them reproduces the old fraction-shaped behavior.
    double low_water_frac = 0.02;
    double high_water_frac = 0.0;
    /// LS_KVXP_MAX_SLOTS_PER_GRANT — bounds the ASK, not the FILL. It caps
    /// how many SLABS one grant may ask for (slabs_for_slots of the capped
    /// slot count); once claimed, the grant takes EVERY slot those slabs
    /// hold, because stranding claimed capacity is exactly the waste the
    /// admissibility bound (44z §3, max_waste) forbids. The two coincide on
    /// every served model (one expert slot spans many slabs); they diverge
    /// only where expert_slot_bytes << slab_bytes, and there the waste
    /// equation wins.
    int max_slots_per_grant = 8;
    int64_t min_tick_interval_us = 200000;  ///< LS_KVXP_TICK_MS (env is ms)
    /// LS_KVXP_GRANT_COOLDOWN_MS — wall-clock window after ANY completed
    /// reclaim during which no NEW grant may be issued. Damps band-edge
    /// thrash: without it a pool sitting near the high-water mark re-grants
    /// the instant a reclaim lands and starts the cycle again. Reclaims are
    /// NEVER delayed by the cooldown — pressure always flows expert -> KV
    /// at full speed; only the opposite direction waits.
    ///
    /// A DURATION, not a tick count: expressed in ticks it silently shortened
    /// whenever LS_KVXP_TICK_MS was lowered, quietly reintroducing the churn
    /// it exists to prevent. The equivalent tick count is derived ONCE, at
    /// resolve_marks() (see ExpertZoneRebalancer::grant_cooldown_ticks_).
    ///
    /// 30 s, raised from ~5 s on evidence. At 5 s a re-grant fired while the
    /// PRESSURE SOURCE was still running — a multi-minute prefill still
    /// growing its KV — and re-took 474 slabs that were reclaimed again
    /// moments later: pure churn, no benefit. 30 s outlasts most prefills'
    /// pressure ramp, so the re-grant lands after the demand that caused the
    /// reclaim has resolved rather than in the middle of it. Grants are
    /// OPPORTUNISTIC by design and lose nothing by waiting; a reclaim is what
    /// has to be prompt. 0 disables the cooldown.
    ///
    /// The cooldown is CHURN DAMPING ONLY — never load-bearing for liveness.
    /// It was once raised as a mitigation for the regrant wedge
    /// (TD-KVXP-RECLAIM-REGRANT-WEDGE); the root cause turned out to live in
    /// the DRAIN (a cache-direct evict bypassing the ELM — see the
    /// evict_expert dep), so no cooldown length could have fixed it, and the
    /// resolved bug places no constraint on this value.
    int64_t grant_cooldown_ms = 30000;

    /// Apply the LS_* env overrides over `base` — the parsed
    /// _internal-kv_expert_rebalance schema section (TD-KVXP-SCHEMA-KNOBS),
    /// or defaults — clamping insane values to the documented ranges.
    /// A SET env var wins EITHER WAY over the config value; an unset one
    /// leaves the config value in force (the kda_state.mapped precedence),
    /// so a measured recipe can pin the policy while an operator can still
    /// flip it in place.  The low/high fractions are FLOORS under the
    /// absolute marks: low >= high is warned (the high floor is subsumed),
    /// never fatal — the hysteresis band is derived absolutely from the
    /// admission transient and always has a checked width (resolve_marks),
    /// which needs the pool geometry and so cannot be checked here.
    static ExpertZoneRebalancerConfig from_config_env(
        const ExpertZoneRebalancerConfig& base);

    /// from_config_env over the built-in defaults (env-only boots, tests).
    static ExpertZoneRebalancerConfig from_env();
};

/// Observability. `granted_*` are CURRENTLY OUTSTANDING (not cumulative);
/// grants / reclaims / refusals are monotonic counters.
struct ExpertZoneRebalancerStats {
    int64_t grants = 0;
    int64_t reclaims = 0;
    int64_t grant_refusals = 0;
    /// THE policy-failure counter: a retryable admission refusal observed
    /// while a grant was outstanding. Non-zero means the low-water mark is
    /// set wrong (the background drain started too late), NOT that the
    /// rebalancer misbehaved.
    int64_t forced_immediate_reclaims = 0;
    /// remove_elastic_zone refused at step (5). The slabs are then LEAKED to
    /// the expert side on purpose: leak-safe beats releasing a region the
    /// cache still believes it owns.
    int64_t zone_remove_failures = 0;
    int granted_slots = 0;   ///< outstanding expert slots handed to the cache
    int granted_slabs = 0;   ///< outstanding kMain slabs lent out
    int draining_zones = 0;  ///< grants mid-reclaim (drain or barrier)
};

class ExpertZoneRebalancer {
public:
    using Deps = ExpertZoneRebalancerDeps;
    using Config = ExpertZoneRebalancerConfig;
    using Stats = ExpertZoneRebalancerStats;

    ExpertZoneRebalancer(Deps deps, Config cfg);

    /// Background maintenance step (DaemonLoop::Deps::background_fn). Rate
    /// limited to one effective pass per Config::min_tick_interval_us on a
    /// steady_clock. Order inside a pass: advance in-flight reclaims first
    /// (capacity returns before anything new is handed out), then the
    /// pressure check, then — only from a comfortable pool — a grant.
    void tick();

    /// Called at the RETRYABLE admission-refusal seam — the dispatcher's
    /// write_error CHOKE POINT for kKvPoolExhausted, so no refusal site can
    /// miss the hook (TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION: the seq_create
    /// bulk-KV site had no per-site call, so a 97k prefill's 189 refusals
    /// never armed the drain and the pool stayed parked just above high
    /// water forever). If any grant is outstanding this is the policy
    /// failure described in the header: it counts the event and arms an
    /// eager drain even though free slabs never crossed the low-water mark.
    ///
    /// DEMAND-AWARE (P-30 step 3): `shortfall_slabs` is how many more slabs
    /// the refused claim needed beyond what was free (0 = unknown). The
    /// eager drain targets free >= free_now + shortfall + one admission's
    /// upfront margin (state transient + per-step growth — the claims that
    /// FOLLOW the refused one in the same admission), draining AS MANY
    /// grants as that takes, not one. The old force-one drain freed a
    /// single ~300-500-slab grant against a demand of thousands, stopped at
    /// `covered >= high`, and the 30 s cooldown then re-granted the freed
    /// span before the client's next retry — an oscillation that permanently
    /// blocked any admission larger than one grant. Repeated refusals
    /// re-arm with a fresh shortfall, so an under-estimate (e.g. the KV
    /// shortfall not seeing a later indexer claim) converges across the
    /// orchestrator's bounded-wait retries.
    void note_pool_pressure_refusal(int gpu_idx, int64_t shortfall_slabs = 0);

    const Stats& stats() const { return stats_; }
    bool enabled() const { return cfg_.enabled; }

    /// EFFECTIVE water marks in SLABS for the current pool (diagnostics +
    /// boot log). "Effective" = after the band-width guard, so
    /// high_water_slabs() may exceed floor(high_water_frac * total).
    int low_water_slabs() const;
    int high_water_slabs() const;
    /// The guard's floor on band width (0 before the marks resolve).
    int64_t min_band_slabs() const { return min_band_slabs_; }
    /// Config::grant_cooldown_ms expressed in effective ticks (0 = disabled).
    /// Derived from the tick interval, so it changes when the interval does —
    /// that is the point: the DURATION is what stays fixed.
    int grant_cooldown_ticks() const { return grant_cooldown_ticks_; }
    /// ADMISSION-DEMAND-AWARE GRANT CAP (P-30 step 3): grants never take the
    /// pool below low + one admission's upfront margin (state transient +
    /// per-step growth), so the boot grant loop cannot park free just above
    /// high water where the FIRST admission's non-deferrable claims no
    /// longer fit (the measured wart: free parked at ~high+6..41 while one
    /// admission needed transient 306 + KV + indexer). Equals high when the
    /// margin is unwired (0) — legacy behavior, verbatim.
    int64_t grant_floor_slabs() const;
    /// Free-slab level the ARMED eager drain is working toward (0 when not
    /// armed). Diagnostics + tests.
    int64_t eager_demand_free_slabs() const { return eager_demand_free_slabs_; }

private:
    /// One outstanding grant: the same slab span on EVERY tp rank, presented
    /// to the cache as one elastic zone per rank.
    struct GrantRec {
        int start_slab = -1;
        int num_slabs = 0;
        int num_slots = 0;
        std::vector<int> zone_ids;  ///< per tp rank (parallel to tp_gpus)
        enum class St {
            kActive,   ///< serving experts; reclaimable
            kDraining, ///< zone refuses reserves; residents being evicted
            kBarrier,  ///< drained + quiesced; waiting on stream events
            kLeaked,   ///< remove_elastic_zone refused — never touched again
        };
        St st = St::kActive;
        /// Parallel to tp_gpus (nullptr where the record failed).
        std::vector<void*> h2d_events;
        std::vector<void*> ffn_events;
    };

    /// Steps (2)-(5) for every grant that is mid-reclaim.
    void advance_reclaims();
    /// Step (2): evict what is evictable right now; true iff every rank's
    /// zone reports drained.
    bool drain_residents(GrantRec& g);
    /// Step (4): h2d + kExpertFfn barrier events on every rank.
    void record_barriers(GrantRec& g);
    /// Step (5) predicate.
    bool barriers_complete(const GrantRec& g) const;
    /// Step (5): destroy events, remove zones, release slabs. Returns false
    /// (leaving the grant kLeaked) when a zone refuses removal.
    bool finish_reclaim(GrantRec& g);
    /// Release every zone/claim of a partially-built grant (rollback).
    void rollback_partial(int start_slab, int num_slabs, int claimed_ranks,
                          const std::vector<int>& zone_ids);

    /// Slabs free on the TIGHTEST rank (a grant needs the span free on all).
    int min_free_slabs() const;
    /// Total slabs of the shared region (rank 0 is representative).
    int total_slabs() const;
    /// Slabs currently held by grants that are already on their way back.
    int slabs_in_reclaim() const;
    bool any_reclaim_in_flight() const;

    void try_grant(int free_slabs, int high);
    void refresh_draining_stat();
    /// Band-width guard. Idempotent: computes min_band_slabs_ and the
    /// EFFECTIVE marks once the pool's slab count is known, widening the
    /// high mark (with a warn) or disabling the rebalancer (with an error)
    /// when the configured band cannot clear the floor. Called from the
    /// constructor and, defensively, from tick() — a pool that is not yet
    /// carved at construction resolves on the first effective tick instead.
    void resolve_marks();

    Deps deps_;
    Config cfg_;
    Stats stats_{};
    /// Grant order; a reclaim pops from the BACK (newest first) — the oldest
    /// grants are the ones the cache has had longest to make useful.
    std::vector<GrantRec> grants_;
    /// Armed by note_pool_pressure_refusal: drain even above low water.
    bool eager_reclaim_ = false;
    /// Free-slab TARGET of the armed eager drain (demand-aware): drains are
    /// started until free + slabs_in_reclaim covers max(high, this). Reset
    /// with eager_reclaim_. 0 while disarmed.
    int64_t eager_demand_free_slabs_ = 0;
    bool first_tick_ = true;
    std::chrono::steady_clock::time_point last_tick_{};

    // ── Band-width guard + grant cooldown ────────────────────────────────
    int64_t min_band_slabs_ = 0;   ///< the checked floor on (high - low)
    int low_slabs_ = 0;            ///< EFFECTIVE marks (post-guard)
    int high_slabs_ = 0;
    bool marks_resolved_ = false;
    /// Config::grant_cooldown_ms expressed in effective ticks, derived ONCE
    /// in resolve_marks(): ms / tick_ms, floored at 1 when the duration is
    /// non-zero so a cooldown can never round away to nothing. A tick
    /// interval of 0 (unbounded tick rate) has no ms-per-tick to divide by
    /// and is treated as 1 ms per tick.
    int grant_cooldown_ticks_ = 0;
    /// Effective ticks remaining before a NEW grant may be issued. Set to
    /// grant_cooldown_ticks_ whenever a reclaim completes.
    int grant_cooldown_left_ = 0;
};

}  // namespace layerstorm::daemon
