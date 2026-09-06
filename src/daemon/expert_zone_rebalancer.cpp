// 44z stage 4: the KV <-> expert zone rebalancer. See the header for the
// asymmetry / hysteresis / reclaim-protocol reasoning this file implements.

#include "daemon/expert_zone_rebalancer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

#include "core/memory/expert_cache.h"
#include "core/memory/page_allocator.h"

namespace layerstorm::daemon {

namespace {

/// Parse a double env var, keeping the default when absent/unparseable.
double env_double(const char* name, double dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    try {
        return std::stod(std::string(v));
    } catch (...) {
        spdlog::warn("44z: {}='{}' is not a number — keeping {}", name, v,
                     dflt);
        return dflt;
    }
}

int64_t env_i64(const char* name, int64_t dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    try {
        return std::stoll(std::string(v));
    } catch (...) {
        spdlog::warn("44z: {}='{}' is not an integer — keeping {}", name, v,
                     dflt);
        return dflt;
    }
}

double clamp_frac(const char* name, double v, double lo, double hi,
                  double dflt) {
    if (!std::isfinite(v)) {
        spdlog::warn("44z: {} not finite — using {}", name, dflt);
        return dflt;
    }
    if (v < lo || v > hi) {
        const double c = std::clamp(v, lo, hi);
        spdlog::warn("44z: {}={} out of [{}, {}] — clamped to {}", name, v, lo,
                     hi, c);
        return c;
    }
    return v;
}

}  // namespace

// ── Config ─────────────────────────────────────────────────────────────────

ExpertZoneRebalancerConfig ExpertZoneRebalancerConfig::from_config_env(
        const ExpertZoneRebalancerConfig& base) {
    // Start from the parsed _internal-kv_expert_rebalance section (or the
    // built-in defaults, which mirror it); every LS_* var below overrides
    // EITHER WAY when set and leaves the config value in force when unset
    // (the kda_state.mapped precedence). The clamps run on the merged value
    // regardless of its source.
    ExpertZoneRebalancerConfig c = base;
    if (const char* v = std::getenv("LS_KV_EXPERT_REBALANCE"))
        c.enabled = (v[0] == '1' && v[1] == '\0');

    c.max_waste = clamp_frac("LS_KVXP_MAX_WASTE",
                             env_double("LS_KVXP_MAX_WASTE", c.max_waste),
                             0.0, 1.0, 0.10);
    // Water marks are FRACTIONS OF TOTAL SLABS. 0.9 is the ceiling: a high
    // mark near 1.0 would mean "grant only from an empty pool", which is
    // never useful and would make the band meaningless.
    c.low_water_frac =
        clamp_frac("LS_KVXP_LOW_FRAC",
                   env_double("LS_KVXP_LOW_FRAC", c.low_water_frac), 0.0, 0.9,
                   0.02);
    c.high_water_frac =
        clamp_frac("LS_KVXP_HIGH_FRAC",
                   env_double("LS_KVXP_HIGH_FRAC", c.high_water_frac), 0.0, 0.9,
                   0.0);

    const int64_t slots =
        env_i64("LS_KVXP_MAX_SLOTS_PER_GRANT", c.max_slots_per_grant);
    c.max_slots_per_grant =
        static_cast<int>(std::clamp<int64_t>(slots, 1, 4096));
    if (slots != c.max_slots_per_grant)
        spdlog::warn("44z: LS_KVXP_MAX_SLOTS_PER_GRANT={} out of [1, 4096] — "
                     "clamped to {}", slots, c.max_slots_per_grant);

    // Tick interval FIRST: the deprecated tick-count cooldown below converts
    // through it.
    const int64_t tick_ms =
        env_i64("LS_KVXP_TICK_MS", c.min_tick_interval_us / 1000);
    const int64_t tick_clamped = std::clamp<int64_t>(tick_ms, 0, 3600000);
    if (tick_ms != tick_clamped)
        spdlog::warn("44z: LS_KVXP_TICK_MS={} out of [0, 3600000] — clamped "
                     "to {}", tick_ms, tick_clamped);
    c.min_tick_interval_us = tick_clamped * 1000;

    // The cooldown is a DURATION. Expressing it as a tick count made it
    // silently shorter whenever LS_KVXP_TICK_MS was lowered — quietly
    // reintroducing exactly the post-reclaim re-grant churn it exists to
    // prevent. The tick count is derived once, at resolve_marks().
    int64_t cooldown_ms = env_i64("LS_KVXP_GRANT_COOLDOWN_MS",
                                  c.grant_cooldown_ms);
    // Deprecated alias, kept only so an existing tick-count setting does not
    // silently become a no-op. Converted through the tick interval.
    if (!std::getenv("LS_KVXP_GRANT_COOLDOWN_MS")) {
        if (const char* legacy = std::getenv("LS_KVXP_GRANT_COOLDOWN_TICKS")) {
            const int64_t ticks = env_i64("LS_KVXP_GRANT_COOLDOWN_TICKS", 0);
            cooldown_ms = ticks * std::max<int64_t>(1, tick_clamped);
            spdlog::warn(
                "44z: LS_KVXP_GRANT_COOLDOWN_TICKS='{}' is DEPRECATED — the "
                "cooldown is now a duration (a tick count shrinks when the "
                "tick interval is lowered). Interpreting {} ticks x {} ms as "
                "{} ms; set LS_KVXP_GRANT_COOLDOWN_MS instead",
                legacy, ticks, std::max<int64_t>(1, tick_clamped), cooldown_ms);
        }
    }
    const int64_t cd_clamped =
        std::clamp<int64_t>(cooldown_ms, 0, 24LL * 3600 * 1000);
    if (cooldown_ms != cd_clamped)
        spdlog::warn("44z: grant cooldown {} ms out of [0, 86400000] — "
                     "clamped to {}", cooldown_ms, cd_clamped);
    c.grant_cooldown_ms = cd_clamped;

    // The band is no longer defined by the fractions — it is derived
    // absolutely from the admission transient and always has a checked
    // width (see resolve_marks), so low >= high can no longer produce a
    // thrashing configuration and is NOT a fatal error. It is still almost
    // certainly a mistake: the high floor is entirely subsumed by the low
    // floor, so setting it had no effect.
    if (c.enabled && c.high_water_frac > 0.0
        && c.low_water_frac >= c.high_water_frac) {
        spdlog::warn(
            "44z: high_water_frac ({}) is at or below low_water_frac "
            "({}) — the high floor is subsumed by the low floor and has no "
            "effect. The hysteresis band is derived from the admission "
            "transient regardless, so this is safe but probably not what you "
            "meant",
            c.high_water_frac, c.low_water_frac);
    }
    return c;
}

ExpertZoneRebalancerConfig ExpertZoneRebalancerConfig::from_env() {
    return from_config_env(ExpertZoneRebalancerConfig{});
}

// ── Construction ───────────────────────────────────────────────────────────

ExpertZoneRebalancer::ExpertZoneRebalancer(Deps deps, Config cfg)
    : deps_(std::move(deps)), cfg_(cfg) {
    deps_.geometry.max_waste = cfg_.max_waste;
    if (cfg_.enabled
        && (!deps_.page_allocator || !deps_.expert_cache
            || deps_.tp_gpus.empty() || !deps_.geometry.valid())) {
        spdlog::error(
            "44z expert-zone rebalancer DISABLED: incomplete deps "
            "(page_allocator={}, expert_cache={}, tp_gpus={}, geometry "
            "slot_bytes={} slab_bytes={})",
            static_cast<const void*>(deps_.page_allocator),
            static_cast<const void*>(deps_.expert_cache), deps_.tp_gpus.size(),
            deps_.geometry.expert_slot_bytes, deps_.geometry.slab_bytes);
        cfg_.enabled = false;
    }
    // Resolve the effective marks now if the pool is already carved (it is,
    // in the engine's boot order and in the tests); tick() retries otherwise.
    resolve_marks();
}

// ── Band-width guard ───────────────────────────────────────────────────────

void ExpertZoneRebalancer::resolve_marks() {
    if (marks_resolved_ || !cfg_.enabled) return;

    // Derive the cooldown's tick count from its DURATION. Done before the
    // not-yet-carved early return below because it depends only on the
    // config, and a reclaim must never find it underived.
    // A tick interval of 0 means "tick every daemon cycle" — there is no
    // ms-per-tick to divide by, so treat a tick as 1 ms; that keeps the
    // mapping deterministic (and is the only degenerate case, since the
    // duration cannot otherwise be expressed in ticks at all).
    const int64_t tick_ms =
        std::max<int64_t>(1, cfg_.min_tick_interval_us / 1000);
    grant_cooldown_ticks_ =
        cfg_.grant_cooldown_ms <= 0
            ? 0  // explicitly disabled
            : static_cast<int>(std::clamp<int64_t>(
                  std::max<int64_t>(1, cfg_.grant_cooldown_ms / tick_ms),
                  1, 100000000));

    const int total = total_slabs();
    if (total <= 0) return;  // pool not carved yet — retry on a later tick

    // ── The marks are ABSOLUTE, not a fraction of the pool ──────────────
    // What must stay free is the in-flight ADMISSION TRANSIENT plus the
    // hysteresis band. That is a property of a single admission, NOT of how
    // big the pool is: a fraction-shaped reserve strands proportionally MORE
    // capacity the emptier the pool is, which is exactly backwards — an idle
    // pool is precisely when the expert cache should be allowed to borrow.
    // (Measured arm: a 0.30 fraction held 4232 slabs / ~1.15 GB / ~62 expert
    // slots hostage; the absolute derivation needs 2312, recovering ~1920
    // slabs / ~28 slots that were reserved for nothing.)
    //
    // The LOW mark covers the engine's own admission parallelism: the reserve
    // is there so that max_concurrent_requests simultaneous admission
    // transients can ALL land without waiting on a reclaim. That is the same
    // worst-case multiplier the KV pool itself is sized by (vram_allocator's
    // max(1, serving.max_concurrent_requests)), so the reserve and the pool
    // agree by construction rather than through a hardcoded constant. At
    // B=1-class configs (concurrency 2) it reduces to the old 2x, so the
    // measured arms stay valid.
    //
    // The fractions survive only as OPTIONAL FLOORS a user can raise:
    //   low  = max(concurrency * transient, ceil(low_frac  * total))
    //   band = max(2 * transient, 2 * grant_slabs(cap, total))
    //   high = max(low + band,    ceil(high_frac * total))
    // Band floor 1 (2x transient): a band narrower than one admission's
    // transient is crossed END TO END by every admission, turning the
    // hysteresis into a per-request grant/reclaim cycle. Band floor 2 (two
    // grants): so one grant can never straddle the band — granting at the
    // top edge would otherwise land free under low water by itself.
    // The pool's own slab count bounds the grant_slabs search: an unbounded
    // run would spin for a very small max_waste, and an S larger than the
    // pool could never be granted anyway.
    const auto ceil_frac = [total](double f) {
        return static_cast<int64_t>(
            std::ceil(f * static_cast<double>(total)));
    };
    const int64_t frac_low = ceil_frac(cfg_.low_water_frac);
    const int64_t frac_high = ceil_frac(cfg_.high_water_frac);
    const int64_t concurrency =
        std::max<int64_t>(1, deps_.max_concurrent_admissions);

    // Size the "could this pool ever grant?" check — and the legacy band
    // floor — at the grant the pool can ACTUALLY produce, not at the raw
    // configured cap. try_grant sizes its ask with best_fit(run) and only
    // then caps it, so a pool smaller than a full-cap grant still hands out
    // smaller admissible grants. Checking grant_slabs(cap, total) alone
    // disabled the rebalancer on any such pool (measured: the GLM-5.2
    // champion's 154-slab kMain pool holds up to 6 admissible slots, yet the
    // 8-slot cap ask needs ~204 slabs and the boot self-disabled —
    // TD-KVXP-FAT-KV-ARM, 2026-09-02).
    const int pool_cap =
        deps_.geometry.best_fit(total, /*base_misalign=*/1).num_slots;
    const int eff_cap = std::min(cfg_.max_slots_per_grant, pool_cap);
    const int64_t two_grants =
        eff_cap >= 1 ? 2 * static_cast<int64_t>(
                           deps_.geometry.grant_slabs(eff_cap, total))
                     : 0;
    if (two_grants <= 0) {
        spdlog::error(
            "44z expert-zone rebalancer DISABLED: no admissible grant of ANY "
            "size up to {} slot(s) fits the {}-slab pool at max_waste "
            "{:.1f}% — there is nothing this rebalancer could ever hand out",
            cfg_.max_slots_per_grant, total, 100.0 * cfg_.max_waste);
        cfg_.enabled = false;
        return;
    }

    // ── TD-KVXP-PER-STEP-FLOOR ──────────────────────────────────────────
    // With the per-step growth term wired, the reserve is the PRINCIPLED
    // floor: per concurrent admission, the non-deferrable upfront state
    // claim plus one incremental growth event's headroom — NOT a whole
    // admission's KV. The band shrinks to two growth events: the bulk
    // prefill excursion the legacy band absorbed is answered by the
    // orchestrator's bounded large-prefill wait on the eager drain instead
    // (an admission near the marks starting a background drain cycle is
    // accepted by design; the 30 s grant cooldown bounds the churn).
    // Unwired (growth 0): the legacy transient-derived marks, verbatim.
    const int64_t growth = std::max<int64_t>(0, deps_.per_step_growth_slabs);
    const bool per_step = growth > 0;
    const int64_t admission_floor =
        concurrency * (deps_.admission_transient_slabs + growth);
    const int64_t band_transient_floor = 2 * deps_.admission_transient_slabs;
    min_band_slabs_ =
        per_step ? 2 * growth
                 : std::max<int64_t>(band_transient_floor, two_grants);

    const int64_t low = std::max<int64_t>(admission_floor, frac_low);
    const int64_t high = std::max<int64_t>(low + min_band_slabs_, frac_high);

    if (high >= total) {
        spdlog::error(
            "44z expert-zone rebalancer DISABLED: the hysteresis marks "
            "cannot fit this pool — low water {} ({} concurrent admissions x "
            "({}-slab state transient + {}-slab per-step growth)) + band {} "
            "puts the high mark at {}, meeting or exceeding the {}-slab "
            "pool. The pool cannot simultaneously hold its own admission "
            "parallelism, a safe band, and a grant — lower "
            "serving.max_concurrent_requests or shrink "
            "LS_KVXP_MAX_SLOTS_PER_GRANT",
            low, concurrency, deps_.admission_transient_slabs, growth,
            min_band_slabs_, high, total);
        cfg_.enabled = false;
        return;
    }

    // Only meaningful when a high floor was actually requested — by default
    // (high_frac 0) the absolute derivation IS the answer, not a widening.
    if (frac_high > 0 && frac_high < low + min_band_slabs_) {
        spdlog::warn(
            "44z: the requested high-water floor {} slabs ({:.3f} of {}) is "
            "NARROWER than the derived band above low water {} — using {} "
            "instead (band floor {} slabs: 2x the per-step growth event "
            "under the per-step floor, or 2x the admission transient / 2 "
            "grants under the legacy derivation)",
            frac_high, cfg_.high_water_frac, total, low, high,
            min_band_slabs_);
    }

    // ADMISSION-DEMAND-AWARE GRANT CAP viability (P-30 step 3): grants stop
    // at max(high, low + one admission's upfront margin). If even that floor
    // meets the pool, no grant could ever be issued without parking free
    // where the next admission's upfront claims cannot land — the exact
    // TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION shape — so disable instead.
    const int64_t upfront_margin =
        deps_.admission_transient_slabs
        + std::max<int64_t>(0, deps_.per_step_growth_slabs);
    if (std::max<int64_t>(high, low + upfront_margin) >= total) {
        spdlog::error(
            "44z expert-zone rebalancer DISABLED: the grant floor cannot fit "
            "this pool — max(high {}, low {} + {}-slab admission upfront "
            "margin) meets or exceeds the {}-slab pool, so every grant would "
            "park free slabs where the next admission's non-deferrable "
            "claims no longer fit",
            high, low, upfront_margin, total);
        cfg_.enabled = false;
        return;
    }

    low_slabs_ = static_cast<int>(low);
    high_slabs_ = static_cast<int>(high);
    marks_resolved_ = true;
    if (per_step) {
        spdlog::info(
            "44z marks resolved (PER-STEP floor, TD-KVXP-PER-STEP-FLOOR): "
            "low {} slabs, band {}, high {} (pool {}). Derivation: low = {} "
            "concurrent admissions x ({}-slab upfront state + {}-slab "
            "per-step growth) = {}; band = 2 x growth = {}. Bulk-prefill "
            "claims above the large-prefill threshold ride the bounded "
            "orchestrator wait instead of a standing reserve. Fraction "
            "floors: low {:.3f}->{} {}, high {:.3f}->{} {}",
            low_slabs_, min_band_slabs_, high_slabs_, total,
            concurrency, deps_.admission_transient_slabs, growth,
            admission_floor, min_band_slabs_,
            cfg_.low_water_frac, frac_low,
            frac_low > admission_floor ? "(ENGAGED)" : "(inactive)",
            cfg_.high_water_frac, frac_high,
            frac_high > low + min_band_slabs_ ? "(ENGAGED)" : "(inactive)");
    } else {
        spdlog::info(
            "44z marks resolved (legacy transient floor — per-step growth "
            "unwired): low {} slabs, band {}, high {} (pool {}). "
            "Derivation: low = {} concurrent admissions x {}-slab "
            "transient = {}; band = max(2x transient = {}, 2 grants of {}) "
            "= {}. Fraction floors: low {:.3f}->{} {}, high {:.3f}->{} {}",
            low_slabs_, min_band_slabs_, high_slabs_, total,
            concurrency, deps_.admission_transient_slabs, admission_floor,
            band_transient_floor, two_grants / 2, min_band_slabs_,
            cfg_.low_water_frac, frac_low,
            frac_low > admission_floor ? "(ENGAGED)" : "(inactive)",
            cfg_.high_water_frac, frac_high,
            frac_high > low + min_band_slabs_ ? "(ENGAGED)" : "(inactive)");
    }
    spdlog::info(
        "44z grant floor {} slabs (max(high {}, low {} + {}-slab admission "
        "upfront margin)) — grants never park free below it "
        "(TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION)",
        grant_floor_slabs(), high_slabs_, low_slabs_, upfront_margin);
}

// ── Pool queries ───────────────────────────────────────────────────────────

int ExpertZoneRebalancer::total_slabs() const {
    if (!deps_.page_allocator || deps_.tp_gpus.empty()) return 0;
    return deps_.page_allocator->kv_fragmentation(deps_.tp_gpus.front())
        .total_slabs;
}

int ExpertZoneRebalancer::min_free_slabs() const {
    if (!deps_.page_allocator || deps_.tp_gpus.empty()) return 0;
    int free = std::numeric_limits<int>::max();
    for (const int g : deps_.tp_gpus)
        free = std::min(free, deps_.page_allocator->kv_fragmentation(g)
                                  .free_slabs);
    return free;
}

// The EFFECTIVE marks. 0 until resolve_marks() runs — the marks are DERIVED
// (admission transient + band), not a formula over the fractions, so there
// is no honest value to synthesize before the pool's slab count is known.
int ExpertZoneRebalancer::low_water_slabs() const {
    return marks_resolved_ ? low_slabs_ : 0;
}

int ExpertZoneRebalancer::high_water_slabs() const {
    return marks_resolved_ ? high_slabs_ : 0;
}

int64_t ExpertZoneRebalancer::grant_floor_slabs() const {
    if (!marks_resolved_) return 0;
    // One admission's UPFRONT margin: the non-deferrable state transient
    // plus one per-step growth event. NOT a whole admission's KV — that
    // stays the eager-drain + bounded-wait path (the per-step floor's whole
    // point). When both terms are unwired (0) this is exactly `high` and
    // the legacy grant budget is unchanged.
    const int64_t margin = deps_.admission_transient_slabs
                           + std::max<int64_t>(0, deps_.per_step_growth_slabs);
    return std::max<int64_t>(high_slabs_,
                             static_cast<int64_t>(low_slabs_) + margin);
}

int ExpertZoneRebalancer::slabs_in_reclaim() const {
    int n = 0;
    for (const auto& g : grants_)
        if (g.st == GrantRec::St::kDraining || g.st == GrantRec::St::kBarrier)
            n += g.num_slabs;
    return n;
}

bool ExpertZoneRebalancer::any_reclaim_in_flight() const {
    for (const auto& g : grants_)
        if (g.st == GrantRec::St::kDraining || g.st == GrantRec::St::kBarrier)
            return true;
    return false;
}

void ExpertZoneRebalancer::refresh_draining_stat() {
    int n = 0;
    for (const auto& g : grants_)
        if (g.st == GrantRec::St::kDraining || g.st == GrantRec::St::kBarrier)
            ++n;
    stats_.draining_zones = n;
}

// ── The tick ───────────────────────────────────────────────────────────────

void ExpertZoneRebalancer::tick() {
    if (!cfg_.enabled) return;

    const auto now = std::chrono::steady_clock::now();
    if (!first_tick_) {
        const auto since = std::chrono::duration_cast<std::chrono::microseconds>(
                               now - last_tick_)
                               .count();
        if (since < cfg_.min_tick_interval_us) return;
    }
    first_tick_ = false;
    last_tick_ = now;

    // The band guard needs the pool's slab count; on a pool that was not yet
    // carved at construction this is where the marks finally resolve (and
    // where an unsafe band disables the rebalancer).
    resolve_marks();
    if (!cfg_.enabled) return;

    // Cooldown is spent per EFFECTIVE tick and gates GRANTS ONLY — decrement
    // before the reclaim pass so a reclaim completing this tick gets its full
    // suppression window, and so a reclaim is never itself delayed.
    if (grant_cooldown_left_ > 0) --grant_cooldown_left_;

    // (A) Capacity already on its way back moves FIRST: a reclaim that can
    // finish this tick raises free_slabs before the pressure check reads it.
    advance_reclaims();

    // (B) Pressure check against the hysteresis band.
    const int total = total_slabs();
    if (total <= 0) return;
    const int free = min_free_slabs();
    const int low = low_water_slabs();
    const int high = high_water_slabs();

    if (free < low || eager_reclaim_) {
        // Drain the NEWEST grants first: the oldest grant has had the longest
        // to become useful cache, and the newest is the one most likely to
        // have been a mistake given the pressure we now see.
        // An ARMED eager reclaim means an admission ACTUALLY failed while we
        // were holding reclaimable capacity. The drain target is then
        // DEMAND-AWARE (P-30 step 3): the refusal reported how much the
        // claim was short, so drain until free covers that demand — the
        // "covered >= high" stop condition is the LOW-WATER trigger's, and
        // letting it veto the response is exactly the shape that permanently
        // blocked a 97k admission (one ~300-slab grant handed back against a
        // ~2,500-slab demand, then re-granted after the cooldown).
        const int64_t target =
            eager_reclaim_
                ? std::max<int64_t>(high, eager_demand_free_slabs_)
                : high;
        int64_t covered = free + slabs_in_reclaim();
        // Even with an unknown (0) shortfall, an armed eager reclaim hands
        // at least one grant back regardless of where the marks sit.
        bool force_one = eager_reclaim_;
        for (auto it = grants_.rbegin(); it != grants_.rend(); ++it) {
            if (covered >= target && !force_one) break;
            if (it->st != GrantRec::St::kActive) continue;
            bool all_begun = true;
            for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
                if (!deps_.expert_cache->begin_drain_elastic_zone(
                        deps_.tp_gpus[r], it->zone_ids[r]))
                    all_begun = false;
            }
            if (!all_begun) {
                // A zone that refuses begin_drain is already draining or gone
                // on that rank; the state machine below still converges (the
                // drain status is the authority), so proceed.
                spdlog::warn(
                    "44z: begin_drain refused on at least one rank for the "
                    "grant at slab {} ({} slabs) — continuing on drain status",
                    it->start_slab, it->num_slabs);
            }
            it->st = GrantRec::St::kDraining;
            covered += it->num_slabs;
            force_one = false;
            spdlog::info(
                "44z reclaim START: draining grant at slab {} ({} slabs, {} "
                "slots) — free {} slabs, low-water {}, high-water {}{}",
                it->start_slab, it->num_slabs, it->num_slots, free, low, high,
                eager_reclaim_ ? " [EAGER: forced by an admission refusal]"
                               : "");
        }
        // Nothing left that could be drained, or the target is covered.
        bool any_active = false;
        for (const auto& g : grants_)
            if (g.st == GrantRec::St::kActive) any_active = true;
        if (!any_active || covered >= target) {
            eager_reclaim_ = false;
            eager_demand_free_slabs_ = 0;
        }
    }

    // (C) Grant — only from a comfortable pool, never while capacity is
    // already moving the other way (handing out slabs mid-reclaim would make
    // the reclaim pointless and re-fragment the region we just cleared),
    // never inside the post-reclaim cooldown (band-edge thrash damping), and
    // never while an eager drain is still armed (an admission is waiting on
    // the very capacity a grant would take).
    if (grant_cooldown_left_ == 0 && !eager_reclaim_
        && !any_reclaim_in_flight() && free > high)
        try_grant(free, high);

    refresh_draining_stat();
}

// ── (C) Grant ──────────────────────────────────────────────────────────────

void ExpertZoneRebalancer::try_grant(int free_slabs, int high) {
    auto* pa = deps_.page_allocator;
    const int rank0 = deps_.tp_gpus.front();

    // Never spend the high-water cushion — nor the ADMISSION-DEMAND floor
    // (P-30 step 3, TD-KVXP-BOOT-OVERGRANT-FIRST-ADMISSION): the budget is
    // what sits ABOVE max(high, low + one admission's upfront margin), so
    // the boot grant loop cannot park free where the first admission's
    // non-deferrable claims no longer fit.
    const int64_t floor_slabs = grant_floor_slabs();
    const int budget = static_cast<int>(
        std::min<int64_t>(free_slabs - floor_slabs,
                          std::numeric_limits<int>::max()));
    if (budget <= 0) return;
    const int run = std::min(pa->largest_free_run(rank0), budget);
    if (run <= 0) return;

    // Sizing runs BEFORE the allocator picks the run, so the base — and with
    // it the pad before slot 0 — is not yet known. slab_bytes is not a
    // 4096-multiple on any served model, so the pad varies per start slab.
    // Both steps therefore reason under the WORST-CASE pad (base_misalign 1
    // => 4095 B): the ask, and then the smallest admissible S that satisfies
    // it. Worst-case-pad monotonicity (see grant_slabs) makes that one check
    // sufficient — the real base can only do better.
    const int ask = std::min(deps_.geometry.best_fit(run, /*misalign=*/1)
                                 .num_slots,
                             cfg_.max_slots_per_grant);
    if (ask < 1) return;
    const int slabs = deps_.geometry.grant_slabs(ask, run);
    if (slabs < 1) {
        ++stats_.grant_refusals;
        spdlog::debug(
            "44z grant refused: no admissible slab count for {} slot(s) "
            "within a {}-slab run (waste bound {:.1f}% unreachable under the "
            "worst-case base pad)",
            ask, run, 100.0 * cfg_.max_waste);
        return;
    }

    // The rebalancer's own margin ON TOP of the INV-4.9f floor: leave the
    // grant floor (high-water cushion + admission-demand margin) behind for
    // KV growth and the next admission's upfront claims.
    const int64_t extra_reserve_pages = floor_slabs * pa->pages_per_slab();

    auto lead = pa->claim_expert_zone(rank0, slabs, extra_reserve_pages);
    if (!lead) {
        ++stats_.grant_refusals;
        spdlog::debug(
            "44z grant refused: claim_expert_zone(gpu {}, {} slabs, reserve "
            "{} pages) declined (headroom or contiguity)",
            rank0, slabs, extra_reserve_pages);
        return;
    }

    // Now the base is known, so the ACTUAL pad is known. max_slots_per_grant
    // bounded the ASK, not the FILL: the grant takes every slot the claimed
    // slabs hold. Re-capping here would strand claimed capacity as WASTE and
    // break the waste equation (44z §3) — on a small-slot geometry
    // (expert_slot_bytes << slab_bytes) that is a third of the region. On the
    // served models a slot spans many slabs, so fill == ask and nothing moves.
    const auto misalign =
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(lead->base)
                             % memory::kExpertZoneSlotAlign);
    const int n = deps_.geometry.slots_in(lead->num_slabs, misalign);
    if (n < 1) {
        pa->release_expert_zone(rank0, lead->start_slab, lead->num_slabs);
        ++stats_.grant_refusals;
        spdlog::debug(
            "44z grant refused: run at slab {} ({} slabs) holds no aligned "
            "slot (base misalign {} B)",
            lead->start_slab, lead->num_slabs, misalign);
        return;
    }
    // Belt and braces. grant_slabs proved admissibility under the WORST-CASE
    // pad and the actual pad can only be smaller, so this is unreachable by
    // monotonicity — which is exactly why it is an ERROR and not a refusal
    // class: reaching it means the geometry contract itself broke.
    const double wf = deps_.geometry.waste_frac(lead->num_slabs, n);
    if (wf > cfg_.max_waste) {
        spdlog::error(
            "44z grant ABORTED: slabs [{}, {}) hold {} slot(s) at waste "
            "{:.3f}% > max {:.3f}% despite worst-case-pad admissibility "
            "(base misalign {} B) — expert-zone geometry contract violated",
            lead->start_slab, lead->start_slab + lead->num_slabs, n,
            100.0 * wf, 100.0 * cfg_.max_waste, misalign);
        pa->release_expert_zone(rank0, lead->start_slab, lead->num_slabs);
        ++stats_.grant_refusals;
        return;
    }

    // Lockstep-by-slab-id across the TP ranks (see the header's TP note).
    std::vector<int> zone_ids;
    int claimed_ranks = 1;
    for (size_t r = 1; r < deps_.tp_gpus.size(); ++r) {
        if (!pa->claim_expert_zone_at(deps_.tp_gpus[r], lead->start_slab,
                                      lead->num_slabs, extra_reserve_pages)) {
            spdlog::debug(
                "44z grant refused: mirror claim of slabs [{}, {}) failed on "
                "gpu {} — rolling the whole grant back (replicated KV claims "
                "by INDEX; a per-rank-divergent span is not usable)",
                lead->start_slab, lead->start_slab + lead->num_slabs,
                deps_.tp_gpus[r]);
            rollback_partial(lead->start_slab, lead->num_slabs, claimed_ranks,
                             zone_ids);
            ++stats_.grant_refusals;
            return;
        }
        ++claimed_ranks;
    }

    const int64_t zone_bytes = deps_.geometry.capacity_bytes(lead->num_slabs);
    for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
        const int g = deps_.tp_gpus[r];
        // claim_expert_zone returns the base only for the CLAIMING rank; the
        // mirrors hold the same slab ids, so their base is derived.
        void* base_g =
            r == 0 ? lead->base
                   : static_cast<void*>(
                         static_cast<char*>(pa->kv_main_base(g))
                         + static_cast<int64_t>(lead->start_slab)
                               * pa->slab_bytes());
        const int zid = deps_.expert_cache->add_elastic_zone(
            g, base_g, zone_bytes, deps_.geometry.slot_stride(), n,
            /*cookie=*/lead->start_slab);
        if (zid < 0) {
            spdlog::warn(
                "44z grant refused: add_elastic_zone failed on gpu {} for "
                "slabs [{}, {}) — rolling the whole grant back",
                g, lead->start_slab, lead->start_slab + lead->num_slabs);
            rollback_partial(lead->start_slab, lead->num_slabs, claimed_ranks,
                             zone_ids);
            ++stats_.grant_refusals;
            return;
        }
        zone_ids.push_back(zid);
    }

    GrantRec rec;
    rec.start_slab = lead->start_slab;
    rec.num_slabs = lead->num_slabs;
    rec.num_slots = n;
    rec.zone_ids = std::move(zone_ids);
    rec.st = GrantRec::St::kActive;
    grants_.push_back(std::move(rec));

    ++stats_.grants;
    stats_.granted_slabs += lead->num_slabs;
    stats_.granted_slots += n;

    std::string gpus;
    for (const int g : deps_.tp_gpus) {
        if (!gpus.empty()) gpus += ",";
        gpus += std::to_string(g);
    }
    spdlog::info(
        "44z GRANT: gpus [{}] slabs [{}, {}) = {} slabs -> {} expert slots "
        "(waste {:.3f}%, max {:.1f}%); free slabs {} -> {} (high-water {})",
        gpus, lead->start_slab, lead->start_slab + lead->num_slabs,
        lead->num_slabs, n,
        100.0 * wf,
        100.0 * cfg_.max_waste, free_slabs, min_free_slabs(), high);
}

void ExpertZoneRebalancer::rollback_partial(int start_slab, int num_slabs,
                                            int claimed_ranks,
                                            const std::vector<int>& zone_ids) {
    for (size_t r = 0; r < zone_ids.size(); ++r)
        deps_.expert_cache->remove_elastic_zone(deps_.tp_gpus[r], zone_ids[r]);
    for (int r = 0; r < claimed_ranks; ++r)
        deps_.page_allocator->release_expert_zone(deps_.tp_gpus[r], start_slab,
                                                  num_slabs);
}

// ── (A) Reclaim protocol, steps (2)-(5) ────────────────────────────────────

void ExpertZoneRebalancer::advance_reclaims() {
    for (size_t i = 0; i < grants_.size();) {
        GrantRec& g = grants_[i];
        if (g.st == GrantRec::St::kActive || g.st == GrantRec::St::kLeaked) {
            ++i;
            continue;
        }

        if (g.st == GrantRec::St::kDraining) {
            // (2) Evict what is evictable NOW. Locked or half-arrived entries
            // just defer the drain to a later tick — never a forced eviction
            // (an in-flight dispatch holds the lock, #90).
            if (!drain_residents(g)) {
                ++i;
                continue;
            }
            // (3) Quiesce — "no MoE KERNEL in flight", NOT "pipeline idle"
            // (the idle reading deadlocks a memory-stalled admission against
            // this very drain; see the header). Checked ONCE at the drained
            // -> barrier edge: it is a proof about in-flight kernels, not a
            // steady-state condition to re-poll.
            if (deps_.moe_quiesced && !deps_.moe_quiesced()) {
                ++i;
                continue;
            }
            // (4) Stream barriers. cancel() does not stop a dispatched DMA.
            record_barriers(g);
            g.st = GrantRec::St::kBarrier;
        }

        // (5) Only now does capacity actually move.
        if (g.st == GrantRec::St::kBarrier) {
            if (!barriers_complete(g)) {
                ++i;
                continue;
            }
            if (!finish_reclaim(g)) {
                ++i;  // kLeaked — held forever, deliberately
                continue;
            }
            grants_.erase(grants_.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
    refresh_draining_stat();
}

bool ExpertZoneRebalancer::drain_residents(GrantRec& g) {
    bool all_drained = true;
    for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
        const int gpu = deps_.tp_gpus[r];
        const int zid = g.zone_ids[r];
        for (const auto key : deps_.expert_cache->elastic_zone_residents(gpu,
                                                                        zid)) {
            const auto* e = deps_.expert_cache->lookup(key, gpu);
            if (!e) continue;
            // READY + UNLOCKED is the whole gate: a half-arrived entry still
            // has an H2D landing in the zone, and a locked one is named by a
            // dispatch that has not finished.
            if (e->lock_count == 0 && e->sub_components_ready == memory::kAll) {
                // Through the LIFECYCLE MANAGER, never the cache directly
                // (INV-ELM-EVICT — see the evict_expert dep comment): a
                // cache-direct evict leaves the ELM's (key,gpu) tier kHot,
                // ensure_resident then never re-fetches the key, and every
                // layer routing it burns its full fetch deadline — the
                // TD-KVXP-RECLAIM-REGRANT-WEDGE silent crawl. A refusal
                // (pending interests / in-flight transfer) defers to a later
                // tick exactly like a locked entry.
                if (deps_.evict_expert)
                    deps_.evict_expert(key, gpu);
                else
                    deps_.expert_cache->evict(key, gpu);
            }
        }
        if (!deps_.expert_cache->elastic_drain_status(gpu, zid).drained)
            all_drained = false;
    }
    return all_drained;
}

void ExpertZoneRebalancer::record_barriers(GrantRec& g) {
    const size_t ranks = deps_.tp_gpus.size();
    g.h2d_events.assign(ranks, nullptr);
    g.ffn_events.assign(ranks, nullptr);
    for (size_t r = 0; r < ranks; ++r) {
        const int gpu = deps_.tp_gpus[r];
        if (deps_.record_h2d_barrier) g.h2d_events[r] = deps_.record_h2d_barrier(gpu);
        if (deps_.record_ffn_barrier) g.ffn_events[r] = deps_.record_ffn_barrier(gpu);
    }
}

bool ExpertZoneRebalancer::barriers_complete(const GrantRec& g) const {
    if (!deps_.event_complete) return true;  // no event surface: nothing to wait on
    for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
        const int gpu = deps_.tp_gpus[r];
        if (r < g.h2d_events.size() && g.h2d_events[r]
            && !deps_.event_complete(gpu, g.h2d_events[r]))
            return false;
        if (r < g.ffn_events.size() && g.ffn_events[r]
            && !deps_.event_complete(gpu, g.ffn_events[r]))
            return false;
    }
    return true;
}

bool ExpertZoneRebalancer::finish_reclaim(GrantRec& g) {
    for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
        const int gpu = deps_.tp_gpus[r];
        if (deps_.destroy_event) {
            if (r < g.h2d_events.size() && g.h2d_events[r])
                deps_.destroy_event(gpu, g.h2d_events[r]);
            if (r < g.ffn_events.size() && g.ffn_events[r])
                deps_.destroy_event(gpu, g.ffn_events[r]);
        }
    }
    g.h2d_events.clear();
    g.ffn_events.clear();

    // The cache must give the region up before the allocator takes it back.
    // If it refuses, the slabs stay lent FOREVER: leaking capacity to the
    // expert side is recoverable (a restart), handing KV a region the cache
    // still indexes is not.
    bool all_removed = true;
    for (size_t r = 0; r < deps_.tp_gpus.size(); ++r) {
        if (!deps_.expert_cache->remove_elastic_zone(deps_.tp_gpus[r],
                                                     g.zone_ids[r])) {
            all_removed = false;
            spdlog::error(
                "44z RECLAIM ABORTED: remove_elastic_zone(gpu {}, zone {}) "
                "refused for slabs [{}, {}) — the {} slabs stay LENT (leaking "
                "capacity to the expert side beats releasing a region the "
                "cache still owns)",
                deps_.tp_gpus[r], g.zone_ids[r], g.start_slab,
                g.start_slab + g.num_slabs, g.num_slabs);
        }
    }
    if (!all_removed) {
        ++stats_.zone_remove_failures;
        g.st = GrantRec::St::kLeaked;
        return false;
    }

    for (const int gpu : deps_.tp_gpus)
        deps_.page_allocator->release_expert_zone(gpu, g.start_slab,
                                                  g.num_slabs);

    ++stats_.reclaims;
    stats_.granted_slabs -= g.num_slabs;
    stats_.granted_slots -= g.num_slots;
    // Band-edge thrash damping: a pool hovering near the high mark would
    // otherwise re-grant the span we just recovered on the very next tick.
    grant_cooldown_left_ = grant_cooldown_ticks_;
    spdlog::info(
        "44z RECLAIM: returned slabs [{}, {}) = {} slabs ({} expert slots) to "
        "KV on {} rank(s); free slabs now {} (low-water {}, high-water {})",
        g.start_slab, g.start_slab + g.num_slabs, g.num_slabs, g.num_slots,
        deps_.tp_gpus.size(), min_free_slabs(), low_water_slabs(),
        high_water_slabs());
    return true;
}

// ── The policy-failure seam ────────────────────────────────────────────────

void ExpertZoneRebalancer::note_pool_pressure_refusal(int gpu_idx,
                                                      int64_t shortfall_slabs) {
    (void)gpu_idx;  // the drain is pool-wide: every rank mirrors the span
    int outstanding = 0;
    for (const auto& g : grants_)
        if (g.st != GrantRec::St::kLeaked) ++outstanding;
    // No grant outstanding means the pool ran dry entirely on its own — the
    // rebalancer had nothing to give back, so this is NOT a policy failure
    // and must not be counted as one.
    if (outstanding == 0) return;

    ++stats_.forced_immediate_reclaims;
    // DEMAND-AWARE target (P-30 step 3): free enough for the refused claim
    // (its reported shortfall) PLUS the same-admission claims that follow it
    // (state transient + one growth event). An unknown shortfall (0) still
    // targets one admission's margin above today's free — at least one grant
    // hands back either way (force_one in the tick). Repeat refusals during
    // the orchestrator's bounded wait re-arm with fresh shortfalls, so an
    // under-estimate converges instead of latching.
    const int64_t total = total_slabs();
    const int64_t margin = deps_.admission_transient_slabs
                           + std::max<int64_t>(0, deps_.per_step_growth_slabs);
    const int64_t demand = std::min<int64_t>(
        total, static_cast<int64_t>(min_free_slabs())
                   + std::max<int64_t>(0, shortfall_slabs) + margin);
    const bool arming = !eager_reclaim_;
    const bool grew = demand > eager_demand_free_slabs_;
    eager_reclaim_ = true;
    if (grew) eager_demand_free_slabs_ = demand;
    if (arming || grew) {
        spdlog::warn(
            "44z POLICY FAILURE: admission refused while {} expert-zone "
            "grants outstanding — eager drain armed toward {} free slabs "
            "(shortfall {} + admission margin {}; forced-immediate reclaim "
            "#{})",
            outstanding, eager_demand_free_slabs_, shortfall_slabs, margin,
            stats_.forced_immediate_reclaims);
    } else {
        // The bounded-wait retry loop re-issues the refused claim every
        // ~100 ms while the drain runs; one WARN per arming (above) is the
        // signal, the repeats are debug.
        spdlog::debug(
            "44z: admission refusal repeated while the eager drain is armed "
            "(target {} free slabs; forced-immediate reclaim #{})",
            eager_demand_free_slabs_, stats_.forced_immediate_reclaims);
    }
}

}  // namespace layerstorm::daemon
