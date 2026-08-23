// ArenaMigrator (M3b) — online self-tuning arena placement via
// swap-through-spare host memcpy relocations. See header.

#include "core/memory/arena_migrator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <spdlog/spdlog.h>

namespace layerstorm::memory {

namespace {

double env_double(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v || !std::isfinite(d) || d <= 0.0) {
        spdlog::warn("ArenaMigrator: ignoring invalid {}='{}' (keeping {})",
                     name, v, fallback);
        return fallback;
    }
    return d;
}

}  // namespace

ArenaMigrator::Config ArenaMigrator::Config::from_env() {
    Config c;
    c.budget_mbps = env_double("LS_ARENA_MIGRATE_MBPS", c.budget_mbps);
    c.half_life_s = env_double("LS_ARENA_MIGRATE_HALF_LIFE_S", c.half_life_s);
    c.ratio = env_double("LS_ARENA_MIGRATE_RATIO", c.ratio);
    c.margin = env_double("LS_ARENA_MIGRATE_MARGIN", c.margin);
    c.min_total = env_double("LS_ARENA_MIGRATE_MIN_TOTAL", c.min_total);
    return c;
}

ArenaMigrator::ArenaMigrator(Config cfg, PinnedExpertArena& arena,
                             std::vector<int> hbm_nodes, uint32_t num_layers,
                             uint32_t num_experts, size_t slot_bytes)
    : cfg_(cfg),
      arena_(arena),
      hbm_nodes_(std::move(hbm_nodes)),
      num_layers_(num_layers),
      num_experts_(num_experts),
      slot_bytes_(slot_bytes),
      freq_(static_cast<size_t>(num_layers) * num_experts, 0.0f) {
    const auto now = std::chrono::steady_clock::now();
    last_decay_ = last_refresh_ = last_log_ = last_budget_ = now;
    last_persist_ = now;
    std::sort(hbm_nodes_.begin(), hbm_nodes_.end());
    copy_thread_ = std::thread([this] { copy_worker_main(); });
}

ArenaMigrator::~ArenaMigrator() {
    {
        std::lock_guard<std::mutex> g(copy_mu_);
        copy_stop_ = true;
    }
    copy_cv_.notify_all();
    copy_thread_.join();  // arena outlives us (engine destroys us first)
    // Final stats — short runs never reach the 60 s periodic log.
    if (stats_.committed || stats_.demotions || stats_.aborted ||
        stats_.spare_steals)
        spdlog::info(
            "arena_migrate[final]: committed={}, demotions={}, aborted={}, "
            "commit_retries={}, skipped_stale={}, spare_steals={}, "
            "copied={:.1f} MiB, fetches_seen={:.0f}",
            stats_.committed, stats_.demotions, stats_.aborted,
            stats_.commit_retries, stats_.skipped_stale, stats_.spare_steals,
            stats_.bytes_copied / 1048576.0, fetches_seen_);
}

// ── Copy worker ──────────────────────────────────────────────────────────────

void ArenaMigrator::copy_worker_main() {
    for (;;) {
        CopyJob job{};
        {
            std::unique_lock<std::mutex> lk(copy_mu_);
            copy_cv_.wait(lk, [this] { return copy_stop_ || !copy_jobs_.empty(); });
            if (copy_stop_ && copy_jobs_.empty()) return;
            job = copy_jobs_.front();
            copy_jobs_.pop_front();
        }
        // Source slot is inflight-pinned (stable bytes), target slot is
        // loading-marked (never reused) — INV-ARENA-MIGRATE.
        std::memcpy(job.dst, job.src, job.bytes);
        {
            std::lock_guard<std::mutex> g(copy_mu_);
            copy_done_.push_back(job.id);
        }
    }
}

uint64_t ArenaMigrator::submit_copy(const void* src, void* dst) {
    const uint64_t id = next_copy_id_++;
    {
        std::lock_guard<std::mutex> g(copy_mu_);
        copy_jobs_.push_back({id, src, dst, slot_bytes_});
    }
    copy_cv_.notify_one();
    stats_.bytes_copied += slot_bytes_;
    return id;
}

bool ArenaMigrator::copy_done(uint64_t id) {
    {
        std::lock_guard<std::mutex> g(copy_mu_);
        if (!copy_done_.empty()) {
            reaped_.insert(reaped_.end(), copy_done_.begin(), copy_done_.end());
            copy_done_.clear();
        }
    }
    auto it = std::find(reaped_.begin(), reaped_.end(), id);
    if (it == reaped_.end()) return false;
    reaped_.erase(it);
    return true;
}

// ── Signal ───────────────────────────────────────────────────────────────────

void ArenaMigrator::on_demand_fetch(ExpertKey key) {
    const size_t i = idx(key);
    if (i < freq_.size()) {
        freq_[i] += 1.0f;
        fetches_seen_ += 1.0;
    }
}

float ArenaMigrator::freq(ExpertKey key) const {
    const size_t i = idx(key);
    return i < freq_.size() ? freq_[i] : 0.0f;
}

void ArenaMigrator::adopt_ema(const float* data, size_t n,
                              double fetches_seen) {
    if (!data || n != freq_.size()) {
        spdlog::warn("ArenaMigrator: adopt_ema shape mismatch ({} != {}) — "
                     "starting cold", n, freq_.size());
        return;
    }
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        freq_[i] = data[i];
        total += data[i];
    }
    stats_.ema_total = total;
    // Adopt-as-is: fetches_seen restores the warm-up-gate state (a mature
    // snapshot re-arms planning immediately; see header).
    fetches_seen_ = std::max(0.0, fetches_seen);
    last_persisted_fetches_ = fetches_seen_;
}

void ArenaMigrator::set_persist_sink(std::function<void()> fn,
                                     double period_s) {
    persist_fn_ = std::move(fn);
    persist_period_s_ = period_s > 0.0 ? period_s : 30.0;
}

void ArenaMigrator::persist_if_due(
        std::chrono::steady_clock::time_point now) {
    if (!persist_fn_) return;
    if (fetches_seen_ == last_persisted_fetches_) return;  // nothing new
    if (std::chrono::duration<double>(now - last_persist_).count() <
        persist_period_s_)
        return;
    last_persist_ = now;
    last_persisted_fetches_ = fetches_seen_;
    persist_fn_();
}

bool ArenaMigrator::is_hbm(int node) const {
    return std::binary_search(hbm_nodes_.begin(), hbm_nodes_.end(), node);
}

void ArenaMigrator::decay_if_due(std::chrono::steady_clock::time_point now) {
    const double dt = std::chrono::duration<double>(now - last_decay_).count();
    if (dt < cfg_.half_life_s / 8.0) return;  // sweep ~8×/half-life
    const float f = static_cast<float>(std::exp2(-dt / cfg_.half_life_s));
    double total = 0.0;
    for (auto& v : freq_) {
        v *= f;
        total += v;
    }
    stats_.ema_total = total;
    last_decay_ = now;
}

// ── Planning ─────────────────────────────────────────────────────────────────

void ArenaMigrator::refresh_queue(std::chrono::steady_clock::time_point now) {
    last_refresh_ = now;
    queue_.clear();
    // Warm-up gate: a cold EMA cannot rank victims (everything reads 0) —
    // early moves would churn on tie-breaks. ~10-20 decode tokens of signal
    // suffice (≈350 demand fetches/token on the keeper regime).
    if (fetches_seen_ < cfg_.min_total) return;

    // Classify residents by node class. (freq desc, key asc) / (freq asc,
    // key asc) orders are fully deterministic.
    std::vector<std::pair<float, ExpertKey>> hot;   // non-HBM residents
    std::vector<std::pair<float, ExpertKey>> cold;  // HBM residents
    for (const auto& [k, node] : arena_.locations()) {
        const size_t i = idx(k);
        const float f = i < freq_.size() ? freq_[i] : 0.0f;
        if (is_hbm(node)) {
            cold.emplace_back(f, k);
        } else if (f > static_cast<float>(cfg_.margin)) {
            hot.emplace_back(f, k);  // only demonstrably hot keys promote
        }
    }
    if (hot.empty()) return;
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    std::sort(cold.begin(), cold.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    // Free HBM capacity first (no victim needed). NOTE: the floating spare
    // never lives on an HBM node (it is minted DDR-side and re-materializes
    // on the promotee's old node), so free HBM slots are genuine targets.
    size_t free_hbm = 0;
    for (int n : hbm_nodes_)
        if (const PinnedNodeArena* a = arena_.node_arena(n))
            free_hbm += a->num_slots() - a->occupied();

    size_t hi = 0, ci = 0;
    while (queue_.size() < cfg_.queue_cap && hi < hot.size() && free_hbm > 0) {
        queue_.push_back({hot[hi].second, kNoEvictedKey});
        ++hi;
        --free_hbm;
    }
    // Then hysteresis-gated swaps against the coldest HBM residents.
    while (queue_.size() < cfg_.queue_cap && hi < hot.size() &&
           ci < cold.size()) {
        if (hot[hi].first <=
            static_cast<float>(cfg_.ratio) * cold[ci].first +
                static_cast<float>(cfg_.margin))
            break;  // sorted: no later pair can pass either
        queue_.push_back({hot[hi].second, cold[ci].second});
        ++hi;
        ++ci;
    }
}

// ── Execution ────────────────────────────────────────────────────────────────

bool ArenaMigrator::ensure_spare() {
    if (spare_node_ >= 0) {
        const PinnedNodeArena* a = arena_.node_arena(spare_node_);
        if (a && a->has_free_slot()) return true;
        spare_node_ = -1;  // stolen by an ambient fill — re-mint
    }
    // Mint the spare: evict the globally coldest DDR-resident key. That key
    // becomes host-non-resident until its next route (usually never — it is
    // the EMA minimum); the engine's J-1 cold path covers it. One-time.
    float best_f = 0.0f;
    ExpertKey best = kNoEvictedKey;
    int best_node = -1;
    for (const auto& [k, node] : arena_.locations()) {
        if (is_hbm(node)) continue;
        const size_t i = idx(k);
        const float f = i < freq_.size() ? freq_[i] : 0.0f;
        if (best == kNoEvictedKey || f < best_f ||
            (f == best_f && k < best)) {
            best_f = f;
            best = k;
            best_node = node;
        }
    }
    if (best == kNoEvictedKey) return false;
    if (!arena_.evict_key(best)) return false;  // pinned right now — retry later
    spare_node_ = best_node;
    ++stats_.spare_steals;
    spdlog::info("arena_migrate: minted spare slot on node {} (evicted cold "
                 "expert ({},{}), ema {:.1f})",
                 spare_node_, best.layer_idx, best.expert_idx, best_f);
    return true;
}

bool ArenaMigrator::begin_relocation(ExpertKey key, int target_node,
                                     Phase copy_phase) {
    void* dst = arena_.migrate_begin(key, target_node);
    if (!dst) return false;
    const void* src = arena_.resolve(key);  // pinned source slot (begin)
    if (!src) {  // unreachable: begin verified ready
        arena_.migrate_abort(key, target_node);
        return false;
    }
    active_copy_ = submit_copy(src, dst);
    phase_ = copy_phase;
    return true;
}

void ArenaMigrator::finish_active() {
    phase_ = Phase::kIdle;
    active_move_ = Move{};
    active_hbm_node_ = -1;
    promote_src_node_ = -1;
    active_copy_ = 0;
}

bool ArenaMigrator::start_next() {
    while (!queue_.empty()) {
        const Move mv = queue_.front();
        queue_.pop_front();

        // Re-verify against reality — the plan may be up to refresh_period_s
        // stale (residency moves under our own migrations).
        const int src_node = arena_.location_node(mv.key);
        if (src_node < 0 || is_hbm(src_node) || !arena_.is_ready(mv.key)) {
            ++stats_.skipped_stale;
            continue;
        }

        if (mv.victim == kNoEvictedKey) {
            // Free-HBM-slot promotion: single relocation, no demote leg.
            int target = -1;
            for (int n : hbm_nodes_)
                if (const PinnedNodeArena* a = arena_.node_arena(n))
                    if (a->num_slots() > a->occupied()) {
                        target = n;
                        break;
                    }
            if (target < 0) {
                ++stats_.skipped_stale;
                continue;
            }
            if (!begin_relocation(mv.key, target, Phase::kPromoteCopy)) {
                ++stats_.skipped_stale;
                continue;
            }
            active_move_ = mv;
            active_hbm_node_ = target;
            promote_src_node_ = src_node;
            tokens_bytes_ -= static_cast<double>(slot_bytes_);
            return true;
        }

        // Swap-via-spare: demote the victim first (it keeps serving from its
        // HBM slot until the commit flip — zero non-residency).
        const int vnode = arena_.location_node(mv.victim);
        if (vnode < 0 || !is_hbm(vnode) || !arena_.is_ready(mv.victim)) {
            ++stats_.skipped_stale;
            continue;
        }
        if (!ensure_spare()) return false;  // nothing evictable — retry later
        if (!begin_relocation(mv.victim, spare_node_, Phase::kDemoteCopy)) {
            ++stats_.skipped_stale;
            continue;
        }
        active_move_ = mv;
        active_hbm_node_ = vnode;
        promote_src_node_ = src_node;
        tokens_bytes_ -= 2.0 * static_cast<double>(slot_bytes_);
        return true;
    }
    return false;
}

void ArenaMigrator::advance(std::chrono::steady_clock::time_point now) {
    switch (phase_) {
        case Phase::kIdle:
            return;
        case Phase::kDemoteCopy:
            if (!copy_done(active_copy_)) return;
            phase_ = Phase::kDemoteCommit;
            commit_wait_start_ = now;
            [[fallthrough]];
        case Phase::kDemoteCommit: {
            const auto r = arena_.migrate_commit(active_move_.victim,
                                                 spare_node_);
            if (r == PinnedExpertArena::MigrateCommit::kRetry) {
                ++stats_.commit_retries;
                if (std::chrono::duration<double>(now - commit_wait_start_)
                        .count() > 10.0) {
                    spdlog::warn("arena_migrate: demote commit of ({},{}) "
                                 "deferred >10 s",
                                 active_move_.victim.layer_idx,
                                 active_move_.victim.expert_idx);
                    commit_wait_start_ = now;
                }
                return;
            }
            ++stats_.demotions;
            spare_node_ = -1;  // consumed; re-materializes after the promote
            // Promote leg into the victim's freed HBM slot.
            if (!begin_relocation(active_move_.key, active_hbm_node_,
                                  Phase::kPromoteCopy)) {
                // Promotee vanished mid-flight (ambient churn) — the freed
                // HBM slot stays free and a later free-slot move claims it.
                ++stats_.aborted;
                finish_active();
            }
            return;
        }
        case Phase::kPromoteCopy:
            if (!copy_done(active_copy_)) return;
            phase_ = Phase::kPromoteCommit;
            commit_wait_start_ = now;
            [[fallthrough]];
        case Phase::kPromoteCommit: {
            const auto r = arena_.migrate_commit(active_move_.key,
                                                 active_hbm_node_);
            if (r == PinnedExpertArena::MigrateCommit::kRetry) {
                ++stats_.commit_retries;
                if (std::chrono::duration<double>(now - commit_wait_start_)
                        .count() > 10.0) {
                    spdlog::warn("arena_migrate: promote commit of ({},{}) "
                                 "deferred >10 s",
                                 active_move_.key.layer_idx,
                                 active_move_.key.expert_idx);
                    commit_wait_start_ = now;
                }
                return;
            }
            ++stats_.committed;
            // The promotee's old slot is now the floating spare.
            spare_node_ = promote_src_node_;
            finish_active();
            return;
        }
    }
}

void ArenaMigrator::log_if_due(std::chrono::steady_clock::time_point now) {
    if (std::chrono::duration<double>(now - last_log_).count() < 60.0) return;
    last_log_ = now;
    if (stats_.committed == last_logged_committed_) return;  // idle: quiet
    spdlog::info(
        "arena_migrate: committed={} (+{}), demotions={}, aborted={}, "
        "commit_retries={}, skipped_stale={}, spare_steals={}, "
        "copied={:.1f} MiB, ema_total={:.0f}",
        stats_.committed, stats_.committed - last_logged_committed_,
        stats_.demotions, stats_.aborted, stats_.commit_retries,
        stats_.skipped_stale, stats_.spare_steals,
        stats_.bytes_copied / 1048576.0, stats_.ema_total);
    last_logged_committed_ = stats_.committed;
}

void ArenaMigrator::tick() {
    // Idle fast path: with nothing in flight and nothing planned, only every
    // 64th call pays for a clock read + due-checks (the daemon cycles are
    // hot, INV-3.4.2).
    if (phase_ == Phase::kIdle && queue_.empty() && (++idle_gate_ & 0x3F))
        return;

    const auto now = std::chrono::steady_clock::now();

    // 1. Advance the (single) in-flight migration.
    advance(now);

    // 2. Signal upkeep.
    decay_if_due(now);

    // 3. Budget accrual (token bucket, capped at 1 s of burst).
    {
        const double dt =
            std::chrono::duration<double>(now - last_budget_).count();
        last_budget_ = now;
        const double rate = cfg_.budget_mbps * 1048576.0;
        tokens_bytes_ = std::min(tokens_bytes_ + rate * dt, rate * 1.0);
    }

    // 4. Plan + start work.
    if (phase_ == Phase::kIdle) {
        if (queue_.empty() &&
            std::chrono::duration<double>(now - last_refresh_).count() >=
                cfg_.refresh_period_s)
            refresh_queue(now);
        if (!queue_.empty() &&
            tokens_bytes_ >= 2.0 * static_cast<double>(slot_bytes_))
            start_next();
    }

    log_if_due(now);
    persist_if_due(now);
}

}  // namespace layerstorm::memory
