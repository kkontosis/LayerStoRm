// REEF orchestrator decision stack — see reef_orch.h. Extracted byte-identical
// from tests/integration/{keeper52,dsp52}_test.cpp (P-25 / dsp52 lineage).

#include "core/gpu_loader/reef_orch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "core/gpu_loader/loader_keeppred.h"
#include "core/gpu_loader/loader_policy.h"
#include "core/memory/pinned_expert_arena.h"

namespace layerstorm::gpu_loader {

// ── INV-REEF-BANK: the shared epoch-latched paired bank seam ────────────────
// See reef_orch.h. One implementation for both consumers; per-consumer
// closure state (each ReefOrch is single-threaded).

void install_arena_bank_seam(ReefOrch& o,
                             const memory::PinnedExpertArena* arena,
                             std::function<int(int)> paired_node) {
    if (!arena) return;  // no arena (streaming-only boots): bank 0 everywhere
    struct SeamState {
        const memory::PinnedExpertArena* arena;
        std::function<int(int)> paired;
        uint64_t epoch = 0;
        std::shared_ptr<const std::unordered_map<memory::ExpertKey, int>> snap;
    };
    auto s = std::make_shared<SeamState>();
    s->arena = arena;
    s->paired = std::move(paired_node);
    o.bank_node_fn = [s](uint32_t layer, uint16_t expert) -> int {
        const uint64_t ep = s->arena->bank_epoch();
        if (ep != s->epoch || !s->snap) {
            s->snap = s->arena->bank_snapshot();
            s->epoch = ep;
        }
        if (!s->snap) return -1;  // pre-first-publish (boot race): bank 0
        const auto it = s->snap->find(
            memory::ExpertKey{layer, expert});
        if (it == s->snap->end()) return -1;
        return s->paired ? s->paired(it->second) : it->second;
    };
}

ReefOrch::ReefOrch(int tp, std::vector<int> caps, LoaderConstants k)
    : K(std::move(k)),
      board(tp, caps.empty() ? 512
                             : *std::max_element(caps.begin(), caps.end())),
      cap(std::move(caps)),
      M(tp) {}

// ── TD-KVXP-CAPACITY-REPUBLISH: live cap refresh (see reef_orch.h) ─────────

void reef_orch_refresh_caps(ReefOrch& o, const std::vector<int>& new_caps,
                            const std::vector<int>& xtp_positions) {
    if (new_caps.size() != o.cap.size()) {
        std::fprintf(stderr,
                     "[reef] refresh_caps: shape mismatch (%zu != %zu) — "
                     "refresh IGNORED, caps left as-is\n",
                     new_caps.size(), o.cap.size());
        return;
    }
    o.cap = new_caps;
    // Re-arm the XTP route caps from the refreshed capacity IFF they are
    // armed. The cap value is the MEASURED stable-zone slot count — the same
    // rule as the boot arming (TD-MOE-PLACEMENT-CAPACITY-CAP), never a tuned
    // constant. A disarmed route_cap (empty) stays disarmed.
    if (!o.route_cap.empty()) {
        o.route_cap.assign(o.cap.size(), -1);
        for (int g : xtp_positions) {
            if (g < 0 || static_cast<size_t>(g) >= o.cap.size()) continue;
            o.route_cap[static_cast<size_t>(g)] =
                o.cap[static_cast<size_t>(g)];
        }
    }
    // Board capacity is a reserve HINT (alloc_slot grows past it) — no
    // resize. Record the refresh in the decision dump so the offline replay
    // (tools/reef_sim) can apply it at the same solve count.
    if (o.decision_dump) {
        std::fprintf(o.decision_dump, "C %llu",
                     static_cast<unsigned long long>(o.solve_count));
        for (size_t g = 0; g < o.cap.size(); ++g)
            std::fprintf(o.decision_dump, " %d", o.cap[g]);
        std::fputc('\n', o.decision_dump);
    }
}

// ── TD-MOE-PLACEMENT-CAPACITY-CAP: post-solve route-capacity repair ────────
// Enforce ReefOrch::route_cap on one solve's assignment (device-index space,
// a[i] in [0,M) or -1 = solver-invalid, emitted as position 0 downstream).
// The solver itself stays frozen; when no cap binds this is a read-only pass
// and the assignment is byte-identical to the uncapped path. When a capped
// device holds more experts than its stable zone can physically fit, its
// overflow MISSES (never resident hits — they already hold slots, and
// assigned hits <= residents <= stable slots, so overflow <= misses) are
// re-homed one at a time onto the headroom device minimizing the solver's
// full evaluate() objective — the same signals the solve ranked by.
// Deterministic: ascending device scan, ascending (i, d) with strict <.
// If no destination has headroom the residual is left in place and counted
// (the serving-level degraded retry stays the net for that physically
// infeasible case).
template <typename Solver>
static void enforce_route_caps(ReefOrch& o, Solver& sv, int n,
                               std::vector<int>& a) {
    if (o.route_cap.empty()) return;
    const int M = o.M;
    int cnt[kMaxDevices] = {0};
    int capj[kMaxDevices];
    bool any_cap = false;
    for (int j = 0; j < M; ++j) {
        const int pos = o.K.devices[static_cast<size_t>(j)].position;
        capj[j] = (pos >= 0 && static_cast<size_t>(pos) < o.route_cap.size())
                      ? o.route_cap[static_cast<size_t>(pos)] : -1;
        if (capj[j] >= 0) any_cap = true;
    }
    if (!any_cap) return;
    for (int i = 0; i < n; ++i) {
        // A solver-invalid entry (-1, r.n < n) makes evaluate() unsafe and
        // the share unknowable — refuse the repair for this route entirely
        // (cannot happen off the two production tiers; defensive only).
        if (a[static_cast<size_t>(i)] < 0 || a[static_cast<size_t>(i)] >= M)
            return;
        ++cnt[a[static_cast<size_t>(i)]];
    }
    bool binds = false;
    for (int j = 0; j < M; ++j)
        if (capj[j] >= 0 && cnt[j] > capj[j]) { binds = true; break; }
    if (!binds) return;
    for (int j = 0; j < M; ++j) {
        while (capj[j] >= 0 && cnt[j] > capj[j]) {
            int best_i = -1, best_d = -1;
            double best_T = 0.0;
            for (int i = 0; i < n; ++i) {
                if (a[static_cast<size_t>(i)] != j) continue;
                if (o.req.cached_at(i, j)) continue;  // hits keep their device
                for (int d = 0; d < M; ++d) {
                    if (d == j) continue;
                    if (capj[d] >= 0 && cnt[d] >= capj[d]) continue;
                    a[static_cast<size_t>(i)] = d;
                    const double T = sv.evaluate(o.K, o.req,
                                                 a.data(), nullptr);
                    a[static_cast<size_t>(i)] = j;
                    if (best_i < 0 || T < best_T) {
                        best_T = T;
                        best_i = i;
                        best_d = d;
                    }
                }
            }
            if (best_i < 0) {
                // Physically infeasible (every headroom-less device capped,
                // or overflow of pure hits — cannot happen when the caps
                // mirror the stable zones). Leave the residual; degrade
                // (+ the serving retry) remains the net.
                if (o.route_cap_residual == 0)
                    std::fprintf(stderr,
                                 "[reef] route-cap residual: device %d over "
                                 "cap %d by %d with no headroom destination "
                                 "(TD-MOE-PLACEMENT-CAPACITY-CAP; degraded "
                                 "finalize + retry remain the net)\n",
                                 j, capj[j], cnt[j] - capj[j]);
                ++o.route_cap_residual;
                break;
            }
            a[static_cast<size_t>(best_i)] = best_d;
            --cnt[j];
            ++cnt[best_d];
            ++o.route_cap_moves;
        }
    }
}

void reef_orch_route(ReefOrch& o, int layer,
                     const std::vector<uint16_t>& topk,
                     std::vector<uint8_t>& assign) {
    using layerstorm::memory::ExpertKey;
    const int M = o.M;
    const int n = static_cast<int>(topk.size());
    // Loud capacity guard (TD-GLM5N-ROUTED-EXPERT-ID-TRUNCATION audit):
    // solver_big is BasicLoaderSolver<kMaxExpertsLarge> with fixed
    // std::array members; its only internal protection is an assert that
    // vanishes under NDEBUG. The header promises callers fail loud beyond
    // kMaxExpertsLarge — enforce it here rather than in every caller.
    if (n > kMaxExpertsLarge) {
        throw std::runtime_error(
            "reef_orch_route: union size " + std::to_string(n) +
            " exceeds kMaxExpertsLarge=" + std::to_string(kMaxExpertsLarge) +
            " (LoaderSolver256 fixed capacity)");
    }
    // 1. Recency + decayed-frequency touch (i-order, mirrors route_moe_by_loader).
    o.board.advance_recency();
    for (int i = 0; i < n; ++i) {
        const ExpertKey key{static_cast<uint32_t>(layer),
                            topk[static_cast<size_t>(i)]};
        float& f = o.freq[key];
        f = static_cast<float>(f * o.freq_mult + 1.0);
        o.board.touch_existing_all(key, o.freq_w * f);
    }
    // EPM-0 keeppred: record THIS visit's routed union as layer L's
    // prediction (the free prev-round-union temporal signal). Consumed by
    // reef_orch_apply. Gated so the un-biased path is untouched.
    if (o.keeppred_w > 0.0) {
        if (static_cast<size_t>(layer) >= o.keeppred_union.size())
            o.keeppred_union.resize(static_cast<size_t>(layer) + 1);
        auto& u = o.keeppred_union[static_cast<size_t>(layer)];
        u.clear();
        for (int i = 0; i < n; ++i) u.insert(topk[static_cast<size_t>(i)]);
    }
    // 2. SolveRequest (mirrors shadow_solve_and_log's req_build + reuse block).
    auto& req = o.req;
    req.num_devices = M;
    req.num_experts = n;
    req.bank_of.resize(static_cast<size_t>(n));
    req.cached.assign(static_cast<size_t>(n) * M, 0);
    req.pinned.assign(static_cast<size_t>(n), -1);
    req.place.assign(static_cast<size_t>(n) * M, 0.0);
    req.evict_cum.clear();
    req.subprep_us.clear();
    req.m2 = false;
    for (int i = 0; i < n; ++i) {
        const ExpertKey key{static_cast<uint32_t>(layer),
                            topk[static_cast<size_t>(i)]};
        const int node = o.bank_node_fn
            ? o.bank_node_fn(static_cast<uint32_t>(layer),
                             topk[static_cast<size_t>(i)])
            : -1;
        int bank = 0;
        for (int b = 0; b < o.K.num_banks; ++b)
            if (o.K.banks[static_cast<size_t>(b)].node == node) { bank = b; break; }
        req.bank_of[static_cast<size_t>(i)] = bank;
        for (int j = 0; j < M; ++j) {
            const int pos = o.K.devices[static_cast<size_t>(j)].position;
            if (o.board.is_resident(pos, key)) {
                req.cached[static_cast<size_t>(i) * M + j] = 1;
                if (req.pinned[static_cast<size_t>(i)] < 0)
                    req.pinned[static_cast<size_t>(i)] = j;  // hits ride residency
            }
        }
    }
    // Reuse place reward: pd[j] = w/(1+age/τ); free slots place for free.
    if (o.reuse_w > 0.0) {
        const double now = o.board.recency_now();
        double pd[kMaxDevices] = {0.0};
        bool any = false;
        for (int j = 0; j < M; ++j) {
            const int pos = o.K.devices[static_cast<size_t>(j)].position;
            if (o.board.resident_count(pos) < o.cap[static_cast<size_t>(pos)])
                continue;  // free slot → free placement
            double age = now - o.board.cheapest_score(pos);
            if (age < 0.0) age = 0.0;
            pd[j] = o.reuse_w / (1.0 + age / o.reuse_tau);
            any = true;
        }
        if (any)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < M; ++j)
                    req.place[static_cast<size_t>(i) * M + j] += pd[j];
    }
    // 3. Solve → target GPU positions. Unions <= 64 use the frozen production
    // solver (decode byte-identity); larger unions the 256-bound
    // instantiation (pinned-greedy tier beyond the exact budgets).
    auto emit = [&](const auto& r, auto& sv) {
        auto& a = o.route_scratch;
        a.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int jj = (i < r.n) ? r.assignment[static_cast<size_t>(i)] : -1;
            a[static_cast<size_t>(i)] = (jj >= 0 && jj < M) ? jj : -1;
        }
        // TD-MOE-PLACEMENT-CAPACITY-CAP: repair over-capacity shares on
        // capped (wave-excluded) positions. No-op (byte-identical) when
        // route_cap is empty or no cap binds. The first engagement is
        // logged once per boot (INV-REEF-CAP acceptance evidence:
        // route_cap_moves > 0 must be observable in the serve log; same
        // stderr channel as the residual warning above).
        const uint64_t cap_moves_before = o.route_cap_moves;
        enforce_route_caps(o, sv, n, a);
        if (cap_moves_before == 0 && o.route_cap_moves > 0)
            std::fprintf(stderr,
                         "[reef] route-cap repair engaged: %llu expert(s) "
                         "re-homed (layer %d, union %d) — over-capacity XTP "
                         "share capped at stable-zone capacity "
                         "(TD-MOE-PLACEMENT-CAPACITY-CAP)\n",
                         static_cast<unsigned long long>(o.route_cap_moves),
                         layer, n);
        for (int i = 0; i < n; ++i) {
            const int jj = a[static_cast<size_t>(i)];
            assign[static_cast<size_t>(i)] = static_cast<uint8_t>(
                jj >= 0 ? o.K.devices[static_cast<size_t>(jj)].position : 0);
        }
    };
    if (n <= kMaxExperts) emit(o.solver.solve(o.K, req), o.solver);
    else                  emit(o.solver_big.solve(o.K, req), o.solver_big);
    ++o.solve_count;
    if (o.decision_dump) {
        std::fprintf(o.decision_dump, "R %d %d |", layer, n);
        for (int i = 0; i < n; ++i)
            std::fprintf(o.decision_dump, " %u:%u:%d:%d",
                         static_cast<unsigned>(topk[static_cast<size_t>(i)]),
                         static_cast<unsigned>(assign[static_cast<size_t>(i)]),
                         req.bank_of[static_cast<size_t>(i)],
                         req.pinned[static_cast<size_t>(i)]);
        std::fputc('\n', o.decision_dump);
    }
}

void reef_orch_apply(ReefOrch& o, int layer, const ReefEntry* entries,
                     ReefVictim* evicts, uint32_t count,
                     const std::vector<uint8_t>* stream_mask,
                     int64_t cur_step) {
    using layerstorm::memory::ExpertKey;
    std::vector<ExpertKey> keys;
    for (int g = 0; g < o.M; ++g) {
        std::vector<uint32_t> miss_ei;
        for (uint32_t i = 0; i < count; ++i) {
            if (entries[i].gpu_idx != g) continue;
            if (stream_mask && (*stream_mask)[i]) continue;  // ELB streamed
            const ExpertKey k{entries[i].layer_idx, entries[i].expert_idx};
            if (!o.board.is_resident(g, k)) miss_ei.push_back(i);
        }
        auto needed_now = [&](const ExpertKey& k) {
            if (k.layer_idx != static_cast<uint32_t>(layer)) return false;
            for (uint32_t i = 0; i < count; ++i)
                if (entries[i].gpu_idx == g && entries[i].expert_idx == k.expert_idx)
                    return true;
            return false;
        };
        const int need_room = static_cast<int>(miss_ei.size());
        size_t vi = 0;
        if (need_room > 0 &&
            o.board.resident_count(g) + need_room > o.cap[static_cast<size_t>(g)]) {
            if (o.evict_oracle) {
                // BELADY / bridge eviction (diagnostic): victim selection off
                // the EXACT recorded future over ALL residents on g (not the
                // cheap tail — allocation is fine for a diagnostic). Fetches
                // NOTHING extra; only reorders evictions. needed_now experts
                // (routed THIS layer) are never candidates.
                std::vector<ExpertKey> res;
                o.board.resident_keys(g, res);
                struct Cand { ExpertKey key; int64_t dist; double score; };
                std::vector<Cand> cands;
                cands.reserve(res.size());
                const int horizon = o.evict_bridge;
                for (const ExpertKey& k : res) {
                    if (needed_now(k)) continue;
                    Cand c;
                    c.key = k;
                    c.score = o.board.score(g, k);
                    if (horizon > 0) {
                        // Bridge: protected iff next demand within `horizon`.
                        const int32_t s = o.evict_oracle->next_use_step(
                            k.layer_idx, static_cast<uint16_t>(k.expert_idx),
                            cur_step, layer);
                        const bool protect =
                            (s >= 0) && (static_cast<int64_t>(s) - cur_step <= horizon);
                        c.dist = protect ? 1 : 0;  // 1 = protected (evict last)
                    } else {
                        c.dist = o.evict_oracle->next_use_dist(
                            k.layer_idx, static_cast<uint16_t>(k.expert_idx),
                            cur_step, layer);
                    }
                    cands.push_back(c);
                }
                if (horizon > 0) {
                    // Two-tier realizable bridge: evict UNPROTECTED first
                    // (base board order = cheapest score), dip into the
                    // protected set only as a last resort — no livelock.
                    std::sort(cands.begin(), cands.end(),
                              [](const Cand& a, const Cand& b) {
                                  if (a.dist != b.dist) return a.dist < b.dist;
                                  if (a.score != b.score) return a.score < b.score;
                                  if (a.key.layer_idx != b.key.layer_idx)
                                      return a.key.layer_idx < b.key.layer_idx;
                                  return a.key.expert_idx < b.key.expert_idx;
                              });
                } else {
                    // Exact Belady: FARTHEST next-use first (never-again
                    // first). Tie-break cheapest board score, then key.
                    std::sort(cands.begin(), cands.end(),
                              [](const Cand& a, const Cand& b) {
                                  if (a.dist != b.dist) return a.dist > b.dist;
                                  if (a.score != b.score) return a.score < b.score;
                                  if (a.key.layer_idx != b.key.layer_idx)
                                      return a.key.layer_idx < b.key.layer_idx;
                                  return a.key.expert_idx < b.key.expert_idx;
                              });
                }
                size_t ci = 0;
                while (o.board.resident_count(g) + need_room
                           > o.cap[static_cast<size_t>(g)] &&
                       vi < miss_ei.size() && ci < cands.size()) {
                    const ExpertKey victim = cands[ci++].key;
                    const uint32_t ei = miss_ei[vi++];
                    evicts[ei].layer_idx  = victim.layer_idx;
                    evicts[ei].expert_idx = static_cast<uint16_t>(victim.expert_idx);
                    evicts[ei].gpu_idx    = static_cast<uint8_t>(g);
                    o.board.evict(g, victim);
                }
            } else if (o.keeppred_w > 0.0) {
                // EPM-0 keeppred: reorder ONLY a bounded cheap-tail window so
                // predicted-reuse (prev-union) victims sink toward KEEP, then
                // evict the cheapest non-needed as before. Bounded bias ⇒
                // still evictable if nothing cheaper remains (no livelock).
                // NO live mallocs: FIXED stack scratch + insertion-sort
                // helper — no full-resident gather, no per-call heap.
                std::array<double,  kKeeppredWindow> kp_eff;
                std::array<uint8_t, kKeeppredWindow> kp_ret;
                std::array<int,     kKeeppredWindow> kp_order;
                // A non-needed victim is guaranteed in the cheapest
                // (need_room + count) prefix; +64 headroom lets retained
                // members defer to unretained ones. Window capped at
                // kKeeppredWindow.
                int want = need_room + static_cast<int>(count) + 64;
                if (want > kKeeppredWindow) want = kKeeppredWindow;
                o.board.cheapest_keys(g, want, keys);
                int nc = static_cast<int>(keys.size());
                if (nc > kKeeppredWindow) nc = kKeeppredWindow;
                for (int z = 0; z < nc; ++z) {
                    kp_eff[static_cast<size_t>(z)] = o.board.score(g, keys[static_cast<size_t>(z)]);
                    const ExpertKey& k = keys[static_cast<size_t>(z)];
                    kp_ret[static_cast<size_t>(z)] =
                        (static_cast<size_t>(k.layer_idx) < o.keeppred_union.size()
                         && o.keeppred_union[k.layer_idx].count(
                                static_cast<uint16_t>(k.expert_idx))) ? 1 : 0;
                }
                keeppred_victim_order(kp_eff.data(), kp_ret.data(), nc,
                                      o.keeppred_w, kp_order.data());
                int oi = 0;
                while (o.board.resident_count(g) + need_room
                           > o.cap[static_cast<size_t>(g)] &&
                       vi < miss_ei.size() && oi < nc) {
                    const ExpertKey victim =
                        keys[static_cast<size_t>(kp_order[static_cast<size_t>(oi++)])];
                    if (needed_now(victim)) continue;
                    const uint32_t ei = miss_ei[vi++];
                    evicts[ei].layer_idx  = victim.layer_idx;
                    evicts[ei].expert_idx = static_cast<uint16_t>(victim.expert_idx);
                    evicts[ei].gpu_idx    = static_cast<uint8_t>(g);
                    o.board.evict(g, victim);
                }
            } else {
            o.board.cheapest_keys(g, need_room + static_cast<int>(count), keys);
            size_t ki = 0;
            while (o.board.resident_count(g) + need_room
                       > o.cap[static_cast<size_t>(g)] &&
                   vi < miss_ei.size() && ki < keys.size()) {
                const ExpertKey victim = keys[ki++];
                if (needed_now(victim)) continue;
                const uint32_t ei = miss_ei[vi++];
                evicts[ei].layer_idx  = victim.layer_idx;
                evicts[ei].expert_idx = static_cast<uint16_t>(victim.expert_idx);
                evicts[ei].gpu_idx    = static_cast<uint8_t>(g);
                o.board.evict(g, victim);
            }
            }
        }
        // Admit the misses at the current clock (on_resident_added semantics).
        for (uint32_t ei : miss_ei) {
            const ExpertKey k{entries[ei].layer_idx, entries[ei].expert_idx};
            o.board.update(g, k, o.board.recency_now());
        }
    }
    if (o.decision_dump) {
        std::fprintf(o.decision_dump, "A %d |", layer);
        for (uint32_t i = 0; i < count; ++i)
            if (evicts[i].expert_idx != 0xFFFF)
                std::fprintf(o.decision_dump, " %u:%u:%u",
                             evicts[i].layer_idx,
                             static_cast<unsigned>(evicts[i].expert_idx),
                             static_cast<unsigned>(evicts[i].gpu_idx));
        std::fputc('\n', o.decision_dump);
    }
}

std::unique_ptr<ReefOrch> make_reef_orch(const std::string& calib_path,
                                         int num_gpus,
                                         std::vector<int> caps,
                                         std::string* desc_out) {
    LoaderConstants K = load(calib_path);  // throws on failure
    if (K.num_devices != num_gpus)
        throw std::runtime_error(
            "reef_orch: calibration devices " + std::to_string(K.num_devices)
            + " != live " + std::to_string(num_gpus)
            + " (" + calib_path + ")");
    auto o = std::make_unique<ReefOrch>(num_gpus, std::move(caps),
                                        std::move(K));
    // Policy knobs (P-26): same source precedence as the engine dispatcher
    // (init_loader_from_env) — explicit env vars > LS_LOADER_POLICY artifact
    // > built-in defaults. A malformed artifact throws (caller decides).
    PolicyParams pol;  // defaults == ReefOrch's
    std::string psrc = "builtin-defaults";
    if (const char* pp = std::getenv("LS_LOADER_POLICY"); pp && *pp) {
        pol = load_policy_params(pp);  // throws on failure
        psrc = std::string("artifact ") + pp;
        if (!pol.epoch.empty()) psrc += " [epoch " + pol.epoch + "]";
    }
    if (apply_policy_env_overrides(pol)) psrc += " + env overrides";
    o->freq_w    = pol.freq_w;
    o->freq_mult = std::exp(-pol.freq_decay);
    o->reuse_w   = pol.reuse_w;
    o->reuse_tau = pol.reuse_tau;
    // EPM-0 keeppred retention-bias weight (default 0 = OFF; byte-identical).
    if (const char* kw = std::getenv("LS_LOADER_KEEPPRED_W"); kw && *kw) {
        o->keeppred_w = std::atof(kw);
        if (o->keeppred_w < 0.0) o->keeppred_w = 0.0;
    }
    // Decision-stream dump probe (TD-BRIDGE-CPP-GAP; see ReefOrch field).
    if (const char* dp = std::getenv("LS_REEF_DECISION_DUMP"); dp && *dp) {
        o->decision_dump = std::fopen(dp, "w");
        if (!o->decision_dump)
            std::fprintf(stderr, "[reef] LS_REEF_DECISION_DUMP unwritable: %s\n",
                         dp);
    }
    if (desc_out) *desc_out = psrc;
    return o;
}

}  // namespace layerstorm::gpu_loader
