#pragma once

// ArenaMigrator (M3b) — ONLINE self-tuning arena placement. Replaces the
// static LS_ARENA_PLACE_FREQ calibration table (arena_placement.h, M3) with
// a live decayed per-(layer,expert) DEMAND-FETCH frequency signal and a
// low-rate background migration worker that promotes the hottest non-HBM
// experts into HBM node slots and demotes the coldest HBM residents out —
// converging toward the M3-static layout organically from ANY starting
// placement (cold legacy fill, a stale static table, or an adopted warm
// store).
//
// SIGNAL (matches tools/loader_xray/freq_table.py semantics): counts actual
// H2D fetch enqueues (misses that put expert bytes on the wire), NOT routed
// activations — hot-in-VRAM experts route often but fetch rarely, and only
// fetched bytes care about host placement. Fed from the ELM fresh-enqueue
// site via Engine wiring. Decay is a periodic exact half-life sweep over a
// dense per-key float array (~78 KiB for GLM-5.2), so a one-time prefill
// burst fades and the ranking tracks the current routing regime
// (TD-ARENA-PLACE-STALENESS: no more regime-frozen tables).
//
// MIGRATION = SWAP-VIA-SPARE, HOST MEMCPY ONLY (INV-ARENA-MIGRATE): a
// promotion is two chained relocations through one floating spare slot —
//   demote:  victim B (coldest HBM) memcpy sB → spare (B keeps serving from
//            sB until the atomic commit flip; ZERO non-residency),
//   promote: A (hottest non-HBM) memcpy sA → the freed sB (A keeps serving
//            from sA until its flip); sA becomes the new spare.
// No expert is EVER host-non-resident during a migration. This is the
// load-bearing property: the first (NVMe-reload) design evicted the victim
// for the read's duration, and a demand miss inside that window delayed an
// arrival → the dispatcher/REEF compute-placement sequence forked → BF16
// numerics diverged (measured: keeper bare-seed probs drift from the first
// migration-coincident step; the churned STORE bytes stayed bit-clean —
// mechanism was residency timing, not corruption). Copies run on a private
// copy worker off the daemon thread, reading a source slot pinned by
// migrate_begin; commits flip atomically on the daemon thread.
//
// SPARE BOOTSTRAP: the arena is full-fit, so the first migration mints the
// spare by evicting the globally coldest (EMA-min) DDR-resident key ONCE —
// that one key stays cold (engine J-1 reloads it on its next route, usually
// never). Rate is capped by a bytes/sec token bucket; memcpy bandwidth is
// node-local RAM, no NVMe involvement.
//
// THREADING: on_demand_fetch()/tick() are daemon-thread-only (INV-ELM-1);
// the copy worker writes ONLY into `loading` target slots and reads ONLY
// inflight-pinned source slots (both pinned against reuse/eviction).
//
// Flag-gated, default OFF: LS_ARENA_PLACE_ONLINE=1. Knobs (env, defaults in
// Config): LS_ARENA_MIGRATE_{MBPS,HALF_LIFE_S,RATIO,MARGIN,MIN_TOTAL}.
// Placement stays PERFORMANCE-ONLY: ON vs OFF trajectories must be
// byte-identical.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/memory/eviction_policy.h"  // ExpertKey
#include "core/memory/pinned_expert_arena.h"

namespace layerstorm::memory {

class ArenaMigrator {
public:
    struct Config {
        double budget_mbps = 400.0;   ///< migration copy cap (2 copies/move)
        double half_life_s = 30.0;    ///< EMA half-life of the fetch signal
        double ratio = 3.0;           ///< promote iff f_hot > ratio*f_cold+margin
        double margin = 10.0;         ///< absolute deadband (fetches)
        double refresh_period_s = 0.5;///< min gap between candidate rescans
        size_t queue_cap = 16;        ///< planned moves per rescan
        double min_total = 3000.0;    ///< warm-up gate: no moves until the
                                      ///< un-decayed fetch count crosses this
                                      ///< (a cold EMA cannot rank victims —
                                      ///< everything reads 0)

        /// Reads LS_ARENA_MIGRATE_{MBPS,HALF_LIFE_S,RATIO,MARGIN,MIN_TOTAL}
        /// (invalid / non-positive values keep the defaults). The ENABLE
        /// decision (LS_ARENA_PLACE_ONLINE=1) is the engine's, not read here.
        static Config from_env();
    };

    /// `hbm_nodes`: the arena nodes whose slots are the hot (high-bandwidth)
    /// placement class — engine passes the node_is_hbm() subset of
    /// arena_nodes(); tests inject any class split. `slot_bytes` = arena
    /// slot stride (one expert).
    ArenaMigrator(Config cfg, PinnedExpertArena& arena,
                  std::vector<int> hbm_nodes, uint32_t num_layers,
                  uint32_t num_experts, size_t slot_bytes);
    ~ArenaMigrator();

    ArenaMigrator(const ArenaMigrator&) = delete;
    ArenaMigrator& operator=(const ArenaMigrator&) = delete;

    /// One demand fetch of `key` was enqueued (H2D on the wire). Daemon
    /// thread only. Cheap: one float add.
    void on_demand_fetch(ExpertKey key);

    /// Drive the migrator: reap copy completions, advance the (single)
    /// in-flight migration, decay the signal, rescan candidates, start the
    /// next move within budget. Daemon thread only; called once per daemon
    /// cycle (near-free when idle).
    void tick();

    /// True while a migration is in flight (any phase). For tests/drain.
    bool active() const { return phase_ != Phase::kIdle; }

    // ── EMA persistence (TD-ARENA-MIGRATE-EMA-PERSIST) ───────────────────

    /// Adopt a persisted EMA snapshot (`n` == num_layers*num_experts floats)
    /// AS-IS — deliberately no wall-time decay-on-adopt: at the default 30 s
    /// half-life ANY cross-boot gap would zero the signal, and what adoption
    /// needs is the RANKING that produced the stored (persisted) layout.
    /// `fetches_seen` restores the warm-up-gate state: a mature snapshot
    /// (>= min_total) re-arms planning immediately (the gate is skipped —
    /// the persisted signal already ranks victims); an immature one keeps
    /// accumulating (a cold-EMA victim pick stays impossible). Shape
    /// mismatch is ignored with a warn (caller pre-validates via
    /// ArenaCache::ema_load). Call before the daemon starts.
    void adopt_ema(const float* data, size_t n, double fetches_seen);

    /// Snapshot access for persistence (daemon thread, or post-join engine).
    const std::vector<float>& freq_table() const { return freq_; }
    double fetches_seen() const { return fetches_seen_; }

    /// Register the persistence sink: invoked from tick() at most every
    /// `period_s` while new fetches accumulate (the engine wires a lambda
    /// that writes the ArenaCache EMA trailer), plus once more from the
    /// engine after daemon join (final snapshot). Call before the daemon
    /// starts.
    void set_persist_sink(std::function<void()> fn, double period_s = 30.0);

    struct Stats {
        uint64_t committed = 0;       ///< promotions flipped to HBM
        uint64_t demotions = 0;       ///< victims flipped out via the spare
        uint64_t aborted = 0;         ///< migrations abandoned mid-flight
        uint64_t commit_retries = 0;  ///< deferred flips (source pinned)
        uint64_t skipped_stale = 0;   ///< queued moves invalidated by reality
        uint64_t spare_steals = 0;    ///< spare slots minted by evicting a
                                      ///< cold key (bootstrap / re-steal)
        uint64_t bytes_copied = 0;    ///< migration memcpy bytes issued
        double   ema_total = 0.0;     ///< current Σ freq (decayed)
    };
    const Stats& stats() const { return stats_; }

    /// Current decayed frequency of `key` (tests/diagnostics).
    float freq(ExpertKey key) const;

private:
    // A promotion runs as two chained spare-mediated relocations:
    //   kDemoteCopy   — victim B: memcpy sB → spare(node S) in flight
    //   kDemoteCommit — flip B → S (retry while sB has demand pins)
    //   kPromoteCopy  — A: memcpy sA → freed sB in flight
    //   kPromoteCommit— flip A → H; sA becomes the new spare
    // A free-HBM-slot move (no victim) skips straight to kPromote*.
    enum class Phase { kIdle, kDemoteCopy, kDemoteCommit, kPromoteCopy,
                       kPromoteCommit };

    struct Move {
        ExpertKey key;                     ///< promotee (non-HBM, hot)
        ExpertKey victim = kNoEvictedKey;  ///< HBM demotee (none: free slot)
    };

    size_t idx(ExpertKey k) const {
        return static_cast<size_t>(k.layer_idx) * num_experts_ + k.expert_idx;
    }
    bool is_hbm(int node) const;
    void decay_if_due(std::chrono::steady_clock::time_point now);
    void refresh_queue(std::chrono::steady_clock::time_point now);
    bool ensure_spare();
    bool start_next();
    bool begin_relocation(ExpertKey key, int target_node, Phase copy_phase);
    void advance(std::chrono::steady_clock::time_point now);
    void finish_active();
    void log_if_due(std::chrono::steady_clock::time_point now);

    // Copy worker (the only non-daemon actor): jobs copy a pinned source
    // slot into a loading target slot; completion ids are reaped in tick().
    struct CopyJob {
        uint64_t id;
        const void* src;
        void* dst;
        size_t bytes;
    };
    void copy_worker_main();
    uint64_t submit_copy(const void* src, void* dst);
    bool copy_done(uint64_t id);

    Config cfg_;
    PinnedExpertArena& arena_;
    std::vector<int> hbm_nodes_;
    uint32_t num_layers_;
    uint32_t num_experts_;
    size_t slot_bytes_;

    std::vector<float> freq_;          ///< dense decayed fetch counts
    double fetches_seen_ = 0.0;        ///< un-decayed total (warm-up gate)
    std::chrono::steady_clock::time_point last_decay_;
    std::chrono::steady_clock::time_point last_refresh_;
    std::chrono::steady_clock::time_point last_log_;
    double tokens_bytes_ = 0.0;        ///< budget bucket
    std::chrono::steady_clock::time_point last_budget_;

    std::deque<Move> queue_;
    Phase phase_ = Phase::kIdle;
    Move active_move_{};
    int active_hbm_node_ = -1;         ///< HBM node receiving the promotee
    int promote_src_node_ = -1;        ///< node the promotee leaves
    uint64_t active_copy_ = 0;         ///< in-flight copy job id
    std::chrono::steady_clock::time_point commit_wait_start_;

    int spare_node_ = -1;              ///< node holding the floating spare
                                       ///< slot (-1: not minted / lost)

    // Copy worker state.
    std::mutex copy_mu_;
    std::condition_variable copy_cv_;
    std::deque<CopyJob> copy_jobs_;
    std::vector<uint64_t> copy_done_;
    bool copy_stop_ = false;
    std::thread copy_thread_;
    uint64_t next_copy_id_ = 1;
    std::vector<uint64_t> reaped_;     // reused scratch

    uint32_t idle_gate_ = 0;           ///< cheap skip counter for idle ticks
    Stats stats_;
    uint64_t last_logged_committed_ = 0;

    // EMA persistence sink (TD-ARENA-MIGRATE-EMA-PERSIST).
    std::function<void()> persist_fn_;
    double persist_period_s_ = 30.0;
    std::chrono::steady_clock::time_point last_persist_;
    double last_persisted_fetches_ = 0.0;
    void persist_if_due(std::chrono::steady_clock::time_point now);
};

}  // namespace layerstorm::memory
