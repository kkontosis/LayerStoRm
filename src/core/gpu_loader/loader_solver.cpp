#include <cmath>
#include "core/gpu_loader/loader_solver.h"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace layerstorm::gpu_loader {

// compute_us is now inline in the header (folds into the hot loops).

namespace {
constexpr double kInf = 1e300;

// Per-expert destination-ingest cost onto device j (0 if cached). Used by the
// standalone evaluate() and to fill the precomputed sx_ table.
inline double subxfer_us(const LoaderConstants& k, const SolveRequest& req, int i, int j) {
  if (req.cached_at(i, j)) return 0.0;
  const TransferCell& cell = k.matrix[req.bank_of[i]][j];
  return cell.rate_us + cell.lat_us;
}

// ── opt 6 helper: residue-aware top-three over device R-values (greedy fast path).
//
// Holds a value per device index (active flag) plus a cached top-three (the three
// largest *active* values, by value, lowest index first on ties — matching the
// index-order std::max accumulation in objective_from_sums). Single-element
// updates (set/remove) are O(1) when the change cannot disturb the top-three and
// rebuild O(M) only when it can (sub-ULP residue updates essentially never do, so
// this is O(1) amortized). max_excluding(d) returns the largest active value over
// indices != d, or 0.0 when none (reproducing the original max's 0.0 seed).
struct Top3 {
  static constexpr int kCap = kMaxDevices;
  double val[kCap];
  bool   act[kCap];
  int    n = 0;            // number of indices (set once via clear())
  // cached top-three (idx, value), value-descending, lowest-index-first on ties.
  int    ti[3];
  double tv[3];
  int    tn = 0;

  void clear(int count) {
    n = count;
    for (int i = 0; i < n; ++i) act[i] = false;
    tn = 0;
  }
  void clear() { clear(kCap); }  // unused count-less form

  // recompute the cached top-three from scratch (O(n)). Tie-break: a value only
  // displaces a slot when strictly greater, so equal values keep the lower index.
  void rebuild() {
    tn = 0;
    ti[0] = ti[1] = ti[2] = -1;
    tv[0] = tv[1] = tv[2] = -kInf;
    for (int i = 0; i < n; ++i) {
      if (!act[i]) continue;
      const double v = val[i];
      if (v > tv[0])      { tv[2]=tv[1]; ti[2]=ti[1]; tv[1]=tv[0]; ti[1]=ti[0]; tv[0]=v; ti[0]=i; }
      else if (v > tv[1]) { tv[2]=tv[1]; ti[2]=ti[1]; tv[1]=v; ti[1]=i; }
      else if (v > tv[2]) { tv[2]=v; ti[2]=i; }
      if (tn < 3) ++tn;
    }
  }

  // True if a change at index `idx` to value `v` (or removal) could alter the
  // cached top-three: idx is already a cached slot, or v reaches the 3rd slot.
  bool affects(int idx, double v) const {
    for (int s = 0; s < tn; ++s) if (ti[s] == idx) return true;
    // would v enter the top-three? (>= the weakest cached slot, or fewer than 3
    // cached and there are inactive candidates promoting it).
    if (tn < 3) return true;
    return v >= tv[2];
  }

  void push(int idx, double v) {  // initial population (caller calls rebuild() after)
    act[idx] = true; val[idx] = v;
  }
  void set(int idx, double v) {
    // affects(idx, v) is true if idx is a cached slot (old membership) OR v could
    // enter the top-three — covering both the "old was top" and "new enters" cases.
    const bool need = affects(idx, v);
    act[idx] = true; val[idx] = v;
    if (need) rebuild();
  }
  void remove(int idx) {
    if (!act[idx]) return;
    bool was_top = false;
    for (int s = 0; s < tn; ++s) if (ti[s] == idx) { was_top = true; break; }
    act[idx] = false;
    if (was_top) rebuild();
  }
  double max_excluding(int d) const {
    for (int s = 0; s < tn; ++s) if (ti[s] != d) return tv[s];
    return 0.0;
  }
};
}  // namespace

// ── opt 1: precompute the N×M subxfer table once per solve (stride kMaxDevices) ──
template <int NMax>
void BasicLoaderSolver<NMax>::fill_sx(const LoaderConstants& k, const SolveRequest& req) {
  for (int i = 0; i < N_; ++i)
    for (int d = 0; d < M_; ++d)
      sx_[static_cast<size_t>(i) * kMaxDevices + d] = subxfer_us(k, req, i, d);
}

template <int NMax>
void BasicLoaderSolver<NMax>::reset_sums() {
  for (int d = 0; d < M_; ++d) { sub_[d] = 0.0; cnt_[d] = 0; nunc_[d] = 0; }
  for (int b = 0; b < B_; ++b) {
    egress_[b] = 0.0;
    bdcnt_[b]  = 0;
    for (int d = 0; d < M_; ++d) bnd_[static_cast<size_t>(b) * kMaxDevices + d] = 0;
  }
  place_acc_ = 0.0;
  prep_acc_  = 0.0;
}

// Place expert i on device d, updating the maintained sums (uses the precomputed
// sx_ table — caller must have called fill_sx).
template <int NMax>
void BasicLoaderSolver<NMax>::apply(const LoaderConstants& k, const SolveRequest& req, int i, int d) {
  cnt_[d] += 1;
  if (!req.subprep_us.empty()) prep_acc_ += req.subprep_us[i];
  if (!req.cached_at(i, d)) {
    sub_[d] += sx_[static_cast<size_t>(i) * kMaxDevices + d];
    const int b = req.bank_of[i];
    egress_[b] += k.banks[b].egress_us;
    int& bd = bnd_[static_cast<size_t>(b) * kMaxDevices + d];
    if (bd++ == 0) bdcnt_[b] += 1;  // device d newly holds an uncached expert from bank b
    nunc_[d] += 1;
    double p = req.place_at(i, d);
    if (req.clamp_place) p = std::max(0.0, p);
    place_acc_ += p;
  }
}

template <int NMax>
void BasicLoaderSolver<NMax>::undo(const LoaderConstants& k, const SolveRequest& req, int i, int d) {
  cnt_[d] -= 1;
  if (!req.subprep_us.empty()) prep_acc_ -= req.subprep_us[i];
  if (!req.cached_at(i, d)) {
    sub_[d] -= sx_[static_cast<size_t>(i) * kMaxDevices + d];
    const int b = req.bank_of[i];
    egress_[b] -= k.banks[b].egress_us;
    int& bd = bnd_[static_cast<size_t>(b) * kMaxDevices + d];
    if (--bd == 0) bdcnt_[b] -= 1;  // device d no longer holds any from bank b
    nunc_[d] -= 1;
    double p = req.place_at(i, d);
    if (req.clamp_place) p = std::max(0.0, p);
    place_acc_ -= p;
  }
}

// T = prep + max(device_makespan, bank_egress) + recon + place + evict, from the
// maintained sums. recon is folded into the device loop; evict skipped when
// absent (opt 3 lean fast path); place is the maintained accumulator.
template <int NMax>
double BasicLoaderSolver<NMax>::objective_from_sums(const LoaderConstants& k, const SolveRequest& req,
                                         Result* out, bool lower_bound) {
  double makespan = 0.0, ov = 0.0, ad = 0.0;
  int participants = 0;
  for (int d = 0; d < M_; ++d) {
    if (cnt_[d] == 0) continue;
    if (m2_active_) {
      // M2v2 exposed wall (SolveRequest::m2 docs): compute stays first-class,
      // credit frozen (m2_arm) ⇒ monotone in sub_/cnt_ ⇒ valid in both modes.
      const double exposed =
          std::max(0.0, req.m2_s[d] * sub_[d] - m2_credit_[d]);
      makespan = std::max(
          makespan,
          req.m2_cpw[d] *
              (req.m2_gc * compute_us(k.devices[d].compute, cnt_[d]) + exposed));
    } else {
      makespan = std::max(makespan, sub_[d] + compute_us(k.devices[d].compute, cnt_[d]));
    }
    ov = std::max(ov, k.devices[d].recon_overhead_us);
    ad += k.devices[d].recon_added_us;
    ++participants;
  }
  // §3.3 contention-aware bank floor: max_b raw_sum_b · (c_b + (1−c_b)/g_b).
  // g_b = bdcnt_[b] (distinct devices with an uncached expert from bank b). At
  // c_b=1 the factor is 1 → exactly the old strict-serial max_b egress_[b].
  //
  // lower_bound mode (DFS internal-node pruning): the factor is NOT monotone in
  // the number of assigned experts when c_b<1 — placing the next expert on a NEW
  // device raises g_b, which can SHRINK the floor below the current partial value.
  // So the partial true floor is not a valid lower bound on any completion. We
  // bound it below by the most-optimistic parallelism g_b=M_ (smallest factor)
  // applied to the current (only-growing) raw_sum_b: factor(c,g_final) ≥
  // factor(c,M_) since g_final ≤ M_, and raw_final ≥ raw_now ⇒ the achievable
  // final floor ≥ raw_now·factor(c,M_). Valid (never over-estimates), and at
  // c_b=1 it equals raw_now (identical to the strict-serial bound).
  double bank_egress = 0.0;
  for (int b = 0; b < B_; ++b) {
    if (egress_[b] <= 0.0) continue;  // no draw → no floor (g_b≥1 guaranteed when >0)
    const int g = lower_bound ? M_ : bdcnt_[b];
    const double eff = egress_[b] * bank_egress_factor(k.banks[b].contention, g);
    bank_egress = std::max(bank_egress, eff);
  }

  const double recon = (participants > 0) ? ov + ad : 0.0;

  double evict = 0.0;
  if (!req.evict_cum.empty()) {
    for (int d = 0; d < M_; ++d) {
      const std::vector<double>& cum = req.evict_cum[d];
      if (cum.empty()) continue;
      const int n = std::min(nunc_[d], static_cast<int>(cum.size()) - 1);
      if (n > 0) evict += cum[n];
    }
  }

  // I8 trainer feedback: global fixed per-layer overhead (additive offsets the
  // per-rate model misses), added outside the max(makespan,bank) term. Default 0.
  const double T = prep_acc_ + std::max(makespan, bank_egress) + recon + place_acc_
                   + evict + k.fixed_overhead_us;
  if (out) {
    out->device_makespan_us = makespan;
    out->bank_egress_us      = bank_egress;
    out->recon_us            = recon;
    out->place_us            = place_acc_;
    out->evict_us            = evict;
    out->prep_us             = prep_acc_;
    out->predicted_us        = T;
  }
  return T;
}

template <int NMax>
double BasicLoaderSolver<NMax>::evaluate(const LoaderConstants& k, const SolveRequest& req,
                              const std::vector<int>& assignment, Result* out) {
  return evaluate(k, req, assignment.data(), out);
}

template <int NMax>
double BasicLoaderSolver<NMax>::evaluate(const LoaderConstants& k, const SolveRequest& req,
                              const int* assignment, Result* out) {
  // Public entry: honor req.m2 verbatim (see SolveRequest docs). Internal tier
  // code calls evaluate_armed() so a fall-through tier never re-arms m2.
  M_ = req.num_devices;
  N_ = req.num_experts;
  m2_arm(req, assignment);
  return evaluate_armed(k, req, assignment, out);
}

template <int NMax>
bool BasicLoaderSolver<NMax>::m2_valid(const SolveRequest& req) const {
  if (!req.m2) return false;
  const size_t m = static_cast<size_t>(req.num_devices);
  return req.m2_s.size() == m && req.m2_o0.size() == m &&
         req.m2_oc.size() == m && req.m2_hsat.size() == m &&
         req.m2_cpw.size() == m;
}

template <int NMax>
void BasicLoaderSolver<NMax>::m2_arm(const SolveRequest& req, const int* assignment) {
  m2_active_ = m2_valid(req);
  if (!m2_active_) return;
  // Frozen credit (SolveRequest::m2 docs): count CACHED experts fixed per
  // device — from the pins when present (solve path pins the hits), else from
  // the supplied assignment (standalone evaluate). Constant during any DFS.
  int hits[kMaxDevices] = {0};
  if (req.pinned.size() == static_cast<size_t>(req.num_experts)) {
    for (int i = 0; i < req.num_experts; ++i) {
      const int p = req.pinned[i];
      if (p >= 0 && p < req.num_devices && req.cached_at(i, p)) ++hits[p];
    }
  } else if (assignment != nullptr) {
    for (int i = 0; i < req.num_experts; ++i) {
      const int d = assignment[i];
      if (d >= 0 && d < req.num_devices && req.cached_at(i, d)) ++hits[d];
    }
  }
  for (int d = 0; d < req.num_devices; ++d) {
    const double hs = std::max(req.m2_hsat[d], 1e-6);
    m2_credit_[d] =
        req.m2_o0[d] + req.m2_oc[d] * hs *
                           (1.0 - std::exp(-static_cast<double>(hits[d]) / hs));
  }
}

template <int NMax>
double BasicLoaderSolver<NMax>::evaluate_armed(const LoaderConstants& k,
                                    const SolveRequest& req,
                                    const int* assignment, Result* out) {
  M_ = req.num_devices;
  N_ = req.num_experts;
  B_ = k.num_banks;
  assert(M_ <= kMaxDevices && B_ <= kMaxBanks && N_ <= NMax);

  reset_sums();
  for (int i = 0; i < N_; ++i) {
    const int d = assignment[i];
    cnt_[d] += 1;
    if (!req.subprep_us.empty()) prep_acc_ += req.subprep_us[i];
    if (!req.cached_at(i, d)) {
      sub_[d] += subxfer_us(k, req, i, d);  // standalone: no sx_ dependency
      const int b = req.bank_of[i];
      egress_[b] += k.banks[b].egress_us;
      int& bd = bnd_[static_cast<size_t>(b) * kMaxDevices + d];
      if (bd++ == 0) bdcnt_[b] += 1;
      nunc_[d] += 1;
      double p = req.place_at(i, d);
      if (req.clamp_place) p = std::max(0.0, p);
      place_acc_ += p;
    }
  }
  return objective_from_sums(k, req, out);
}

// ── opt 4+5: exact via DFS branch-and-bound. Incremental sums along the path;
// the partial objective over assigned experts is a valid lower bound (every term
// is monotone non-decreasing as experts are added, given clamp0 place + convex
// evict), so prune when it reaches the incumbent. Devices explored in index
// order; the leaf updates the incumbent with its TRUE objective whenever it
// passes the lb gate (historical semantics — NOT strict-improving when any
// bank has contention<1, see the seeding note below).
//
// The §3.3 contention bank floor (c<1) is NOT monotone in #assigned (spreading to
// a new device can shrink it), so the prune uses the lower_bound form of the bank
// term (optimistic g_b=M_); the leaf uses the TRUE objective for the incumbent.
//
// Since the residual-bound work (2026-07-18) bnb() routes through the SAME
// dfs_free machinery as bnb_pinned, with an IDENTITY free list: expert i is free
// slot i, so the exploration order, apply/undo sequence, leaf gate values, and
// incumbent updates are exactly the historical dfs()'s — plus the suffix
// residual bounds (leaf-gate-valid, see fill_free_residuals). The all-miss
// decode layers (zero pins → this tier) measured ~1.4 ms/solve naive. ──
template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::bnb(const LoaderConstants& k, const SolveRequest& req) {
  nfree_ = N_;
  for (int i = 0; i < N_; ++i) free_idx_[i] = i;
  reset_sums();
  fill_free_residuals(k, req);
  for (int i = 0; i < N_; ++i) { cur_a_[i] = 0; best_a_[i] = 0; }
  best_T_ = seed_incumbent(k, req);  // sound under the strict-improving leaf
  dfs_free(k, req, 0);
  Result r;
  r.n = N_;
  for (int i = 0; i < N_; ++i) r.assignment[i] = best_a_[i];
  r.exact = true;
  evaluate_armed(k, req, r.assignment.data(), &r);
  return r;
}

// ── Residual admissible lower bounds for the pinned DFS ──────────────────────
// The plain node bound (partial objective over the assigned prefix) is weak at
// shallow depths: the unassigned free experts contribute NOTHING, so a
// miss-heavy decode layer (nfree=8) exploded to ~4^8 visited nodes and
// ~1.4 ms/solve (measured, LS_LOADER_MACH_PROF 2026-07-18 — 400 ms of a 100-tok
// keeper in nfree=8 solves alone). These suffix tables price the REMAINING free
// experts' unavoidable cost so the root bound already carries most of the final
// objective, collapsing the search. SOUNDNESS CRITERION (leaf-gate validity —
// see the seeding note above bnb_pinned): every term lower-bounds each
// descendant LEAF's GATE value (the lb-mode objective the leaf is pruned
// against — optimistic bank factor(c, M), same composition), also in floating
// point (slack note below). A pruned subtree therefore contains only leaves
// that would have FAILED their own gate against the then-current best_T_, so
// the historical gate-passer sequence — and the returned assignment — is
// BIT-IDENTICAL to the unpruned search.
//
// Terms (suffix over free-list positions fi..nfree_):
//   * bank draw: an expert uncached on EVERY device adds its bank's egress_us
//     wherever it is placed — guaranteed extra draw on that bank. Applied with
//     the same optimistic contention factor(c_b, M_) the lower_bound mode uses.
//     Guarded per-bank on egress_us >= 0 (monotone-growth argument needs it).
//   * makespan solo floor: expert i placed on d makes R_d >= sx[i][d] +
//     compute(d, 1) (sums only grow; compute monotone) — so final makespan >=
//     min_d of that, and >= the max over remaining free experts. Guarded on all
//     compute coefficients and free-row sx being >= 0 (monotonicity).
//   * place: each free expert eventually adds >= min_d of its (clamped) place
//     contribution (0 when cached at d — apply() skips place for hits).
//   * prep: subprep is device-independent — the suffix sum is exact.
//
// FP admissibility: the DFS accumulates these quantities in a different
// addition order than the suffix precompute, so exact-real lower bounds can
// exceed the accumulated double by O(n·eps) ulps. lb_slack() shrinks every
// residual by 1e-12 relative — >=100x the worst-case n<=64 summation error
// (still >=30x at the NMax=256 instantiation's n<=256) —
// making the bounds valid over the reachable FP values while costing a
// negligible amount of pruning strength (~1e-9 us on a ~1e3 us objective).
namespace {
inline double lb_slack(double x) {
  return x >= 0.0 ? x * (1.0 - 1e-12) : x * (1.0 + 1e-12);
}
}  // namespace

template <int NMax>
void BasicLoaderSolver<NMax>::fill_free_residuals(const LoaderConstants& k, const SolveRequest& req) {
  // Guards: the makespan floor needs monotone nonneg compute + nonneg sx over
  // the free rows; disabled (zeroed) otherwise. Bank residuals are guarded
  // per-bank on egress_us >= 0 at the accumulation site.
  // Under m2 the legacy-form makespan floors (res_ms_/res_minsx_/avg-load) are
  // NOT admissible: the saturating credit can absorb a future miss's transfer
  // entirely, so min_d(sx+compute_1) over-prices its true m2 increment. Zero
  // them (ms_ok=false); bank/place/prep residuals price unchanged terms and
  // stay active. The m2 prefix objective itself remains a valid bound.
  bool ms_ok = !m2_active_;
  for (int d = 0; d < M_ && ms_ok; ++d) {
    if (k.devices[d].compute.a_us < 0.0 || k.devices[d].compute.b_us < 0.0) {
      ms_ok = false;
      break;
    }
  }
  for (int f = 0; f < nfree_ && ms_ok; ++f) {
    const int i = free_idx_[f];
    for (int d = 0; d < M_; ++d) {
      if (sx_[static_cast<size_t>(i) * kMaxDevices + d] < 0.0) {
        ms_ok = false;
        break;
      }
    }
  }
  // Hoist the per-solve-constant pieces of the node bound (bitwise-identical
  // values — pure functions of per-solve inputs, just precomputed once).
  for (int b = 0; b < B_; ++b)
    fbank_[b] = bank_egress_factor(k.banks[b].contention, M_);
  for (int d = 0; d < M_; ++d)
    for (int c = 0; c <= N_; ++c)
      comp_[static_cast<size_t>(d) * (NMax + 1) + c] =
          compute_us(k.devices[d].compute, c);
  res_avg_ok_ = ms_ok;  // same guards: nonneg compute (R_d >= sub_d) + nonneg sx
  // Avg-floor compute addend: every still-unassigned expert adds >= min_d a_us
  // of linear compute to SOME device (compute(c+1) >= compute(c) + a under the
  // nonneg guards; the batch ceil term only grows).
  res_min_a_ = 0.0;
  if (ms_ok && M_ > 0) {
    res_min_a_ = kInf;
    for (int d = 0; d < M_; ++d)
      res_min_a_ = std::min(res_min_a_, k.devices[d].compute.a_us);
    if (res_min_a_ < 0.0) res_min_a_ = 0.0;
    res_min_a_ = lb_slack(res_min_a_);
  }
  const bool have_prep = !req.subprep_us.empty();
  // Zero terminal row (fi == nfree_: nothing remains).
  for (int b = 0; b < B_; ++b)
    res_bank_[static_cast<size_t>(nfree_) * kMaxBanks + b] = 0.0;
  res_ms_[nfree_] = 0.0;
  res_place_[nfree_] = 0.0;
  res_prep_[nfree_] = 0.0;
  res_minsx_[nfree_] = 0.0;
  for (int f = nfree_ - 1; f >= 0; --f) {
    const int i = free_idx_[f];
    double* rb = &res_bank_[static_cast<size_t>(f) * kMaxBanks];
    const double* rb1 = &res_bank_[static_cast<size_t>(f + 1) * kMaxBanks];
    for (int b = 0; b < B_; ++b) rb[b] = rb1[b];
    bool unc_all = true;
    for (int d = 0; d < M_; ++d) {
      if (req.cached_at(i, d)) {
        unc_all = false;
        break;
      }
    }
    const int bi = req.bank_of[i];
    if (unc_all && bi >= 0 && bi < B_ && k.banks[bi].egress_us >= 0.0)
      rb[bi] += k.banks[bi].egress_us;
    double mi = 0.0, msx = 0.0;
    if (ms_ok) {
      mi = kInf;
      msx = kInf;
      for (int d = 0; d < M_; ++d) {
        const double s = sx_[static_cast<size_t>(i) * kMaxDevices + d];
        const double v = s + compute_us(k.devices[d].compute, 1);
        if (v < mi) mi = v;
        if (s < msx) msx = s;
      }
      if (mi < 0.0) mi = 0.0;    // defensive (guards should preclude)
      if (msx < 0.0) msx = 0.0;
    }
    res_ms_[f] = std::max(res_ms_[f + 1], mi);
    res_minsx_[f] = res_minsx_[f + 1] + msx;
    double pi = kInf;
    for (int d = 0; d < M_; ++d) {
      double v = 0.0;
      if (!req.cached_at(i, d)) {
        v = req.place_at(i, d);
        if (req.clamp_place) v = std::max(0.0, v);
      }
      if (v < pi) pi = v;
    }
    res_place_[f] = res_place_[f + 1] + pi;
    res_prep_[f] = res_prep_[f + 1] + (have_prep ? req.subprep_us[i] : 0.0);
  }
  // One slack pass (FP admissibility, see header note).
  for (int f = 0; f < nfree_; ++f) {
    double* rb = &res_bank_[static_cast<size_t>(f) * kMaxBanks];
    for (int b = 0; b < B_; ++b) rb[b] = lb_slack(rb[b]);
    res_ms_[f] = lb_slack(res_ms_[f]);
    res_place_[f] = lb_slack(res_place_[f]);
    res_prep_[f] = lb_slack(res_prep_[f]);
    res_minsx_[f] = lb_slack(res_minsx_[f]);
  }
  // Symmetry chains: prev_same_[f] = latest g < f whose free expert is
  // indistinguishable from f's under EVERY objective input (bitwise-equal sx /
  // cached / place rows, same bank, equal subprep). An all-miss decode layer
  // (8 identical fetches) collapses from 4^8 orderings to the sorted multisets.
  const bool have_prep2 = !req.subprep_us.empty();
  for (int f = 0; f < nfree_; ++f) {
    prev_same_[f] = -1;
    const int i = free_idx_[f];
    for (int g = f - 1; g >= 0; --g) {
      const int jx = free_idx_[g];
      if (req.bank_of[i] != req.bank_of[jx]) continue;
      bool same = true;
      for (int d = 0; d < M_ && same; ++d) {
        if (req.cached_at(i, d) != req.cached_at(jx, d) ||
            sx_[static_cast<size_t>(i) * kMaxDevices + d] !=
                sx_[static_cast<size_t>(jx) * kMaxDevices + d] ||
            req.place_at(i, d) != req.place_at(jx, d))
          same = false;
      }
      if (same && have_prep2 && req.subprep_us[i] != req.subprep_us[jx])
        same = false;
      if (same) {
        prev_same_[f] = g;
        break;
      }
    }
  }
}

// Node bound = the partial objective (same shape/composition order as
// objective_from_sums' lower_bound mode) with the suffix residuals folded in.
// Component-wise each piece is <= its value at ANY completing leaf, and the
// final composition mirrors objective_from_sums' association order, so by
// FP monotonicity of +/max the bound never exceeds any leaf objective.
template <int NMax>
double BasicLoaderSolver<NMax>::lb_free(const LoaderConstants& k, const SolveRequest& req,
                             int fi) const {
  double makespan = 0.0, ov = 0.0, ad = 0.0, tot = 0.0;
  int participants = 0;
  for (int d = 0; d < M_; ++d) {
    tot += sub_[d];
    if (cnt_[d] == 0) continue;
    double rd;
    if (m2_active_) {
      // M2 prefix makespan (same form as objective_from_sums; credit frozen ⇒
      // monotone ⇒ a valid completion bound). res_avg_ok_ is false under m2 so
      // the tot/avg floor never consumes this branch's rd.
      const double exposed =
          std::max(0.0, req.m2_s[d] * sub_[d] - m2_credit_[d]);
      rd = req.m2_cpw[d] *
           (req.m2_gc *
                comp_[static_cast<size_t>(d) * (NMax + 1) + cnt_[d]] +
            exposed);
    } else {
      rd = sub_[d] + comp_[static_cast<size_t>(d) * (NMax + 1) + cnt_[d]];
    }
    makespan = std::max(makespan, rd);
    tot += rd - sub_[d];  // fold this device's current compute into the total
    ov = std::max(ov, k.devices[d].recon_overhead_us);
    ad += k.devices[d].recon_added_us;
    ++participants;
  }
  if (res_ms_[fi] > makespan) makespan = res_ms_[fi];
  if (res_avg_ok_) {
    // Average-load floor: total device time (current ingest+compute, plus the
    // remaining experts' guaranteed minima: min-device sx and min linear
    // compute) spread perfectly over M devices still lower-bounds max_d R_d
    // (all terms nonneg under the guards). Slacked for FP admissibility.
    const double avg = lb_slack(
        (tot + res_minsx_[fi] + res_min_a_ * static_cast<double>(nfree_ - fi)) /
        M_);
    if (avg > makespan) makespan = avg;
  }
  double bank_egress = 0.0;
  const double* rb = &res_bank_[static_cast<size_t>(fi) * kMaxBanks];
  for (int b = 0; b < B_; ++b) {
    const double draw = egress_[b] + rb[b];
    if (draw <= 0.0) continue;
    const double eff = draw * fbank_[b];  // == bank_egress_factor(c_b, M_)
    bank_egress = std::max(bank_egress, eff);
  }
  const double recon = (participants > 0) ? ov + ad : 0.0;
  double evict = 0.0;
  if (!req.evict_cum.empty()) {
    for (int d = 0; d < M_; ++d) {
      const std::vector<double>& cum = req.evict_cum[d];
      if (cum.empty()) continue;
      const int n = std::min(nunc_[d], static_cast<int>(cum.size()) - 1);
      if (n > 0) evict += cum[n];
    }
  }
  return (prep_acc_ + res_prep_[fi]) + std::max(makespan, bank_egress) + recon +
         (place_acc_ + res_place_[fi]) + evict + k.fixed_overhead_us;
}

// ── Greedy warm-start incumbent VALUE (threshold-only) ───────────────────────
// From the current pinned state, place each free expert (index order) on the
// device minimizing the full incremental objective, take the completed
// assignment's TRUE objective T_g, undo, and start best_T_ just ABOVE it. The
// DFS prunes the bad region immediately yet still reaches the canonical
// optimal leaf: every ancestor of that leaf has lb <= optimum <= T_g < seed,
// so it is never pruned before the first strict-< update. SOUND because the
// leaf update is STRICT-IMPROVING (P-25 LASTPASSER re-baseline, 2026-07-18 —
// the historical non-strict "last gate-passer" update made any seeding flip
// near-tie results; see INV-LOADER-SOLVER-GATE for the history): the returned
// assignment is the DFS-first leaf achieving the minimum objective value,
// independent of the starting threshold as long as it exceeds the optimum.
// The seed assignment itself is NEVER returned (best_a_ starts at the cur_a_
// default and an update is guaranteed: optimum <= T_g < seed). The margin
// (1e-9 relative + 1e-9 us absolute) keeps the threshold strictly above ANY
// FP re-association of T_g the DFS could compute for the same assignment.
// Uses order_ as scratch (only dp_prefix uses it, never concurrently).
template <int NMax>
double BasicLoaderSolver<NMax>::seed_incumbent(const LoaderConstants& k, const SolveRequest& req) {
  for (int f = 0; f < nfree_; ++f) {
    const int i = free_idx_[f];
    int bd = 0;
    double bt = kInf;
    for (int d = 0; d < M_; ++d) {
      apply(k, req, i, d);
      const double t = objective_from_sums(k, req, nullptr);
      undo(k, req, i, d);
      if (t < bt) { bt = t; bd = d; }
    }
    order_[f] = bd;
    apply(k, req, i, bd);
  }
  const double t_seed = objective_from_sums(k, req, nullptr);
  for (int f = nfree_ - 1; f >= 0; --f) undo(k, req, free_idx_[f], order_[f]);
  return t_seed * (1.0 + 1e-9) + 1e-9;
}

// ── Pinned exact solve: req.pinned experts are pre-applied to the incremental
// sums as FIXED assignments; the DFS branches only over free_idx_[0..nfree_).
// The pinned prefix makes the level-0 lower bound already include the pinned
// contribution, so pruning is at least as strong as bnb()'s — and the suffix
// residuals (fill_free_residuals/lb_free) price the unassigned remainder, so
// miss-heavy layers prune near the root instead of exploding ~M^nfree. Exact
// over the restricted domain (the caller's pinning policy defines what
// "optimal" means — for the dispatcher that is "hits stay on their resident
// device", the ACT execution semantics). Determinism matches bnb() (index
// order, identical leaf gates/updates); residual bounds leaf-gate-valid. ──
template <int NMax>
void BasicLoaderSolver<NMax>::dfs_free(const LoaderConstants& k, const SolveRequest& req, int fi) {
  const double lb = lb_free(k, req, fi);
  if (lb >= best_T_) return;                                // prune
  if (fi == nfree_) {                                       // complete: TRUE objective,
    // STRICT-IMPROVING update (P-25 LASTPASSER re-baseline): the returned
    // assignment is the DFS-first minimal leaf — canonical argmin semantics,
    // which is what makes the seeded threshold and the symmetry restriction
    // below sound. (The historical unconditional overwrite returned the LAST
    // gate-passing leaf; fingerprints recorded before 2026-07-18T2 encode it.)
    const double T = objective_from_sums(k, req, nullptr, /*lower_bound=*/false);
    if (T < best_T_) {
      best_T_ = T;
      best_a_ = cur_a_;
    }
    return;
  }
  const int i = free_idx_[fi];
  // Symmetry: an expert identical to an earlier free expert (bitwise-equal
  // objective inputs, see prev_same_ in fill_free_residuals) only explores
  // devices >= that expert's current device. In exact arithmetic the sorted
  // representative of each permutation class contains the optimum; under FP
  // re-association a swapped permutation can differ by ulps, so the returned
  // leaf is the SORTED representative of the ulp-tie class — deterministic
  // and canonical under the strict-improving update (part of the same
  // re-baseline).
  const int p = prev_same_[fi];
  const int d0 = (p >= 0) ? cur_a_[free_idx_[p]] : 0;
  for (int d = d0; d < M_; ++d) {
    cur_a_[i] = d;
    apply(k, req, i, d);
    dfs_free(k, req, fi + 1);
    undo(k, req, i, d);
  }
}

template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::bnb_pinned(const LoaderConstants& k, const SolveRequest& req) {
  reset_sums();
  fill_free_residuals(k, req);
  for (int i = 0; i < N_; ++i) {
    const int p = req.pinned[i];
    if (p >= 0) {
      cur_a_[i] = p;
      apply(k, req, i, p);
    } else {
      cur_a_[i] = 0;
    }
  }
  best_a_ = cur_a_;  // placeholder; the DFS is guaranteed to update it
                     // (optimum < seed threshold, see seed_incumbent)
  best_T_ = seed_incumbent(k, req);
  dfs_free(k, req, 0);
  Result r;
  r.n = N_;
  for (int i = 0; i < N_; ++i) r.assignment[i] = best_a_[i];
  r.exact = true;
  evaluate_armed(k, req, r.assignment.data(), &r);
  return r;
}

// Optimally place C experts ex[0..C-1] (C<=kMaxC) onto M devices given the
// per-device initial load (init_sub_/init_cnt_), writing into a[ex[.]].
template <int NMax>
void BasicLoaderSolver<NMax>::dp_core(const LoaderConstants& k, const SolveRequest& req, const int* ex, int C,
                           int* a) {
  assert(C <= kMaxC);
  const int M    = req.num_devices;
  const int full = (1 << C) - 1;

  // Subset popcount, computed once (device-independent) — avoids the per-subset
  // __builtin_popcount, which is a software libcall (__popcountdi2) without
  // -mpopcnt. pc[T] = pc[T without low bit] + 1.
  int pc[kDpStates];
  pc[0] = 0;
  for (int T = 1; T <= full; ++T) pc[T] = pc[T & (T - 1)] + 1;

  for (int d = 0; d < M; ++d) {
    // per-device compute over the C+1 possible group sizes (init_cnt + popcount),
    // instead of calling compute_us per subset.
    double comp[kMaxC + 1];
    for (int p = 0; p <= C; ++p) comp[p] = compute_us(k.devices[d].compute, init_cnt_[d] + p);
    sacc_[0] = 0.0;
    for (int T = 1; T <= full; ++T) {
      const int low = T & (-T);
      const int b   = __builtin_ctz(static_cast<unsigned>(low));
      sacc_[T] = sacc_[T ^ low] + sx_[static_cast<size_t>(ex[b]) * kMaxDevices + d];
    }
    for (int T = 0; T <= full; ++T)
      rd_[d * kDpStates + T] = init_sub_[d] + sacc_[T] + comp[pc[T]];
  }

  double* cur = dp_.data();
  double* nxt = ndp_.data();
  for (int T = 0; T <= full; ++T) cur[T] = kInf;
  cur[0] = 0.0;
  for (int d = 0; d < M; ++d) {
    const double* rd_d = &rd_[d * kDpStates];        // hoist per-device bases
    int*          ch_d = &choice_[d * kDpStates];
    for (int S = 0; S <= full; ++S) {
      double best     = kInf;  // accumulate the min in a register; store once below
      int    best_sub = 0;
      int sub = S;
      while (true) {  // all submasks of S (incl. 0 = device d idle, contributes its baseline)
        // unreachable cur[.] == kInf; max() propagates it and it never wins the
        // min, so the explicit reachability branch is dead work — dropped.
        const double cand = std::max(cur[S ^ sub], rd_d[sub]);
        if (cand < best) { best = cand; best_sub = sub; }
        if (sub == 0) break;
        sub = (sub - 1) & S;
      }
      nxt[S]   = best;
      ch_d[S]  = best_sub;
    }
    std::swap(cur, nxt);  // opt 5: ping-pong instead of copy
  }

  int S = full;
  for (int d = M - 1; d >= 0; --d) {
    const int T = choice_[d * kDpStates + S];
    for (int b = 0; b < C; ++b)
      if (T & (1 << b)) a[ex[b]] = d;
    S ^= T;
  }
}

template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::dp_full(const LoaderConstants& k, const SolveRequest& req) {
  const int N = req.num_experts;
  assert(N <= kMaxC);
  for (int d = 0; d < req.num_devices; ++d) { init_sub_[d] = 0.0; init_cnt_[d] = 0; }
  int ex[kMaxC];
  for (int i = 0; i < N; ++i) ex[i] = i;
  Result r;
  r.n = N;
  for (int i = 0; i < N; ++i) r.assignment[i] = 0;
  dp_core(k, req, ex, N, r.assignment.data());
  // The subset-partition DP optimizes makespan only; place/evict (consequence) and
  // the §3.3 contention bank floor (assignment-dependent when any c_b<1) are not in
  // the DP recursion, so the result is only exact when none of those are active.
  r.exact = req.place.empty() && req.evict_cum.empty() && !any_parallel_bank(k);
  evaluate_armed(k, req, r.assignment.data(), &r);
  return r;
}

// True iff any bank's egress channel is modeled as (partially) parallel
// (contention<1) — which makes the §3.3 bank floor assignment-dependent and the
// makespan-only DP no longer provably exact.
template <int NMax>
bool BasicLoaderSolver<NMax>::any_parallel_bank(const LoaderConstants& k) const {
  for (int b = 0; b < B_; ++b)
    if (k.banks[b].contention < 1.0) return true;
  return false;
}

template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::dp_prefix(const LoaderConstants& k, const SolveRequest& req) {
  const int M = req.num_devices;
  const int N = req.num_experts;
  const int C = kMaxC;
  const int F = N - C;  // frozen (costliest)
  assert(N > C && N <= NMax);

  // Route 3 (LPT): order experts by descending "cost when alone".
  for (int i = 0; i < N; ++i) {
    double best = kInf;
    for (int d = 0; d < M; ++d)
      best = std::min(best, sx_[static_cast<size_t>(i) * kMaxDevices + d] + compute_us(k.devices[d].compute, 1));
    solo_[i] = best;
  }
  std::iota(order_.begin(), order_.begin() + N, 0);
  std::sort(order_.begin(), order_.begin() + N, [&](int x, int y) { return solo_[x] > solo_[y]; });

  Result r;
  r.n = N;
  for (int i = 0; i < N; ++i) r.assignment[i] = 0;
  int* a = r.assignment.data();
  for (int d = 0; d < M; ++d) { init_sub_[d] = 0.0; init_cnt_[d] = 0; }
  for (int kf = 0; kf < F; ++kf) {  // LPT-place the costliest F onto the least-loaded device
    const int i = order_[kf];
    int best_d = 0;
    double best = kInf;
    for (int d = 0; d < M; ++d) {
      const double newR = init_sub_[d] + sx_[static_cast<size_t>(i) * kMaxDevices + d] +
                          compute_us(k.devices[d].compute, init_cnt_[d] + 1);
      if (newR < best) { best = newR; best_d = d; }
    }
    a[i] = best_d;
    init_sub_[best_d] += sx_[static_cast<size_t>(i) * kMaxDevices + best_d];
    init_cnt_[best_d] += 1;
  }

  int ex[kMaxC];
  for (int c = 0; c < C; ++c) ex[c] = order_[F + c];
  dp_core(k, req, ex, C, a);

  r.exact = false;
  evaluate_armed(k, req, a, &r);

  // "1 greedy + DP": never worse than greedy.
  Result g = solve_greedy(k, req);
  return (r.predicted_us <= g.predicted_us) ? r : g;
}

template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::solve(const LoaderConstants& k, const SolveRequest& req) {
  M_ = req.num_devices;
  N_ = req.num_experts;
  B_ = k.num_banks;
  assert(M_ <= kMaxDevices && B_ <= kMaxBanks && N_ <= NMax);

  if (M_ <= 0 || N_ <= 0) {
    Result r;
    r.n = std::max(N_, 0);
    for (int i = 0; i < r.n; ++i) r.assignment[i] = 0;
    evaluate_armed(k, req, r.assignment.data(), &r);
    return r;
  }

  fill_sx(k, req);  // opt 1
  m2_active_ = false;  // m2 arms ONLY on the pinned tier (SolveRequest docs)

  // Tier 0: pinned-domain exact B&B — branch only over the free experts when the
  // caller restricted the domain (req.pinned) and the free subspace fits the
  // enumeration budget. Decode (N=8, M=4, ~2 misses free) drops 4^8 -> 4^2 here.
  // Out-of-range pins or an over-budget free subspace fall through to the
  // unrestricted tiers below (pins ignored — documented in the header).
  if (req.pinned.size() == static_cast<size_t>(N_)) {
    nfree_ = 0;
    bool valid = false, in_range = true;
    for (int i = 0; i < N_; ++i) {
      const int p = req.pinned[i];
      if (p >= M_) { in_range = false; break; }
      if (p < 0) free_idx_[nfree_++] = i;
      else valid = true;                 // at least one pin — worth the tier
    }
    if (valid && in_range) {
      uint64_t fspace = 1;
      bool fwithin = true;
      for (int f = 0; f < nfree_; ++f) {
        if (fspace > kMaxEnumerations / static_cast<uint64_t>(M_)) { fwithin = false; break; }
        fspace *= static_cast<uint64_t>(M_);
      }
      if (fwithin) {
        m2_arm(req, nullptr);  // pins present -> credit from pinned hits
        return bnb_pinned(k, req);
      }
    }
  }

  uint64_t space = 1;
  bool within = true;
  for (int i = 0; i < N_; ++i) {
    if (space > kMaxEnumerations / static_cast<uint64_t>(M_)) { within = false; break; }
    space *= static_cast<uint64_t>(M_);
  }
  if (within) return bnb(k, req);          // tier 1: exact DFS B&B
  if (N_ <= kMaxC) return dp_full(k, req);  // tier 2: exact subset-partition DP
  if constexpr (NMax == kMaxExperts) {
    return dp_prefix(k, req);               // tier 3: greedy-prefix (route 3)
  } else {
    // Tier 3' (large instantiation ONLY — the 64 path above is frozen): beyond
    // the exact/DP budgets a batched-verify union routes to the bounded greedy
    // (pins honored verbatim, misses admitted on the full incremental
    // objective). See the header tier note.
    return greedy_pinned(k, req);
  }
}

// ── Tier 3' (NMax > 64): bounded pinned greedy for large batched-verify unions.
//
// Pin hits first: every req.pinned[i] >= 0 (the dispatcher pins cache HITS to
// their resident device — the ACT executes them there anyway) is applied
// verbatim. Then the free experts (the misses) are admitted greedily in INDEX
// order: each tries every device with apply / objective_from_sums / undo — the
// FULL incremental objective, i.e. exactly the signals the exact tiers and the
// REEF orchestrator rank by (per-device fetch-load/makespan balance, the §3.3
// contention bank floor, the place[] reuse reward = victim age from the
// EvictScoreBoard, evict_cum, prep) — and commits to the strict minimum
// (d == 0 || t < best: the lowest device index wins exact ties). Deterministic:
// fixed index order + strict tie-break, no RNG, no data-dependent reordering.
// O(nfree·M) objective evaluations, each O(M+B) — order-µs at the live shape
// (union≈128, ~30 misses, M=4; see GpuLoaderSolver256.LiveShapeMicrobench).
//
// m2 arming mirrors the pinned exact tier: pins present -> credit frozen from
// the pinned hits (m2_arm(req, nullptr); m2_valid gates inside). Without pins
// the tier degenerates to the plain forward greedy over all experts.
template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::greedy_pinned(
    const LoaderConstants& k, const SolveRequest& req) {
  const bool have_pins = req.pinned.size() == static_cast<size_t>(N_);
  if (have_pins) m2_arm(req, nullptr);  // same rule as the pinned exact tier
  reset_sums();
  nfree_ = 0;
  for (int i = 0; i < N_; ++i) {
    int p = have_pins ? req.pinned[i] : -1;
    if (p >= M_) p = -1;  // defensive: out-of-range pin -> treat as free
    if (p >= 0) {
      cur_a_[i] = p;
      apply(k, req, i, p);
    } else {
      cur_a_[i] = 0;
      free_idx_[nfree_++] = i;
    }
  }
  for (int f = 0; f < nfree_; ++f) {
    const int i = free_idx_[f];
    int best_d = 0;
    double best = 0.0;
    for (int d = 0; d < M_; ++d) {
      apply(k, req, i, d);
      const double t = objective_from_sums(k, req, nullptr);
      undo(k, req, i, d);
      if (d == 0 || t < best) { best = t; best_d = d; }
    }
    cur_a_[i] = best_d;
    apply(k, req, i, best_d);  // commit
  }
  Result r;
  r.n = N_;
  for (int i = 0; i < N_; ++i) r.assignment[i] = cur_a_[i];
  r.exact = false;
  evaluate_armed(k, req, r.assignment.data(), &r);
  return r;
}

// ── opt 6: near-O(1)-per-candidate incremental greedy.
//
// The dominant objective on the common (P1) path is T = prep + max(makespan,
// floor), where makespan = max_d R_d (R_d = sub_[d] + compute_us(cnt_[d])) and
// floor = max_b raw_sum_b · factor(c_b, g_b) (the §3.3 contention-aware bank
// floor; raw_sum_b = egress_[b], g_b = bdcnt_[b] = distinct devices with an
// uncached expert from bank b). Makespan keeps its O(1) top-three tracking
// (newR folded with the background max over e≠d).
//
// FLOOR: the contention factor makes the bank term assignment-DEPENDENT via g_b,
// so the old "top-two over egress_" fast path (which assumed per-bank egress was
// assignment-invariant) is no longer valid — the candidate device choice can bump
// g_{bank_i}, changing its effective floor. We recompute the floor as
// max_b egress_[b]·factor(c_b, bdcnt_[b]) per candidate: O(B) with B≤16, a few
// tens of ns, negligible against the ~µs solve. egress_[b]/bdcnt_[b] are the SAME
// doubles/ints the reference full-recompute objective reads (apply/undo maintain
// both), so the result stays bit-identical to the reference greedy. At c_b=1 the
// factor is 1 and this reduces to the old max_b egress_[b].
//
// The consequence terms (place / evict / recon) break the makespan bookkeeping;
// when ANY of them is present we fall back to the original full-recompute path
// (apply / objective_from_sums / undo). ──
template <int NMax>
typename BasicLoaderSolver<NMax>::Result BasicLoaderSolver<NMax>::solve_greedy(const LoaderConstants& k, const SolveRequest& req) {
  M_ = req.num_devices;
  N_ = req.num_experts;
  B_ = k.num_banks;
  assert(M_ <= kMaxDevices && B_ <= kMaxBanks && N_ <= NMax);
  fill_sx(k, req);

  // Detect the consequence terms once. place: non-empty place vector; evict:
  // non-empty evict_cum; recon: any device carries a recon overhead/added term.
  bool has_consequence = !req.place.empty() || !req.evict_cum.empty();
  if (!has_consequence) {
    for (int d = 0; d < M_; ++d) {
      if (k.devices[d].recon_overhead_us != 0.0 || k.devices[d].recon_added_us != 0.0) {
        has_consequence = true;
        break;
      }
    }
  }

  reset_sums();
  Result r;
  r.n = N_;
  for (int i = 0; i < N_; ++i) r.assignment[i] = 0;
  int* a = r.assignment.data();

  if (has_consequence) {
    // Slow but fully general fall-back (the original opt-2 path).
    for (int i = 0; i < N_; ++i) {
      int best_d = 0;
      double best = 0.0;
      for (int d = 0; d < M_; ++d) {
        apply(k, req, i, d);
        const double t = objective_from_sums(k, req, nullptr);
        undo(k, req, i, d);
        if (d == 0 || t < best) { best = t; best_d = d; }
      }
      a[i] = best_d;
      apply(k, req, i, best_d);  // commit
    }
    r.exact = false;
    evaluate_armed(k, req, a, &r);
    return r;
  }

  // ── fast path. apply/undo are kept per candidate so sub_/egress_/bdcnt_ evolve
  // through the EXACT same FP/integer add-then-subtract sequence as the reference
  // full-recompute greedy (the resulting sub-ULP residue can flip ties, so
  // reproducing it is required for bit-identity). The win is replacing the
  // original O(M) makespan scan with O(1) tracker lookups; the floor is an O(B)
  // recompute (B≤16) because the §3.3 contention factor made it assignment-
  // dependent (see header note).
  //
  // FLOOR (bank egress): recomputed per candidate as
  // max_b egress_[b]·factor(c_b, bdcnt_[b]) over the LIVE (apply'd) state. Both
  // egress_[b] and bdcnt_[b] are maintained by apply/undo and read exactly as the
  // reference objective reads them. At c_b=1 factor=1 → max_b egress_[b].
  //
  // MAKESPAN (device R): each candidate d touches a DIFFERENT device, so by the
  // time candidate d is scored, devices 0..d-1 carry residue from their own
  // candidate apply/undo. The original reads those residued sub_[e] live. We
  // reproduce this with a "background" top-three over device R: device e holds its
  // committed R until it has been a candidate, after which it switches to its
  // residued R (a single sub-ULP update). bg_excl(d) = the top-two of that
  // background excluding device d (top-three lets us drop one entry). The candidate
  // device d's own (live, apply'd) R is folded in separately as newR.
  Top3 bg;  // background device-R tracker (residue-aware)

  for (int i = 0; i < N_; ++i) {
    // Build the all-committed background for this expert (devices with cnt>0).
    bg.clear(M_);
    for (int d = 0; d < M_; ++d) {
      if (cnt_[d] == 0) continue;
      bg.push(d, sub_[d] + compute_us(k.devices[d].compute, cnt_[d]));
    }
    bg.rebuild();

    int best_d = 0;
    double best = 0.0;
    for (int d = 0; d < M_; ++d) {
      apply(k, req, i, d);  // sub_[d] += sx; egress_[bank_i]+=eg; bdcnt_ updated; cnt_[d]++
      const double newR     = sub_[d] + compute_us(k.devices[d].compute, cnt_[d]);
      const double otherR   = bg.max_excluding(d);  // background max over e != d
      const double makespan = std::max(newR, otherR);
      // §3.3 contention-aware bank floor over the live state (O(B), B≤16).
      double floor_v = 0.0;
      for (int b = 0; b < B_; ++b) {
        if (egress_[b] <= 0.0) continue;
        floor_v = std::max(floor_v,
                           egress_[b] * bank_egress_factor(k.banks[b].contention, bdcnt_[b]));
      }
      // prep_acc_ is read live (maintained by apply/undo, including its FP residue):
      // although d-independent in exact arithmetic, prep + x rounding can map two
      // candidates' max(...) to the same double, which changes the strict-< tie.
      const double t        = prep_acc_ + std::max(makespan, floor_v);
      undo(k, req, i, d);   // restores sub_[d]/egress_[bank_i]/bdcnt_/cnt_[d] (leaves residue)
      // Switch device d in the background to its now-residued R: every later
      // candidate must see this exactly as the original does.
      if (cnt_[d] == 0) bg.remove(d);  // never participated -> stays out
      else              bg.set(d, sub_[d] + compute_us(k.devices[d].compute, cnt_[d]));
      if (d == 0 || t < best) { best = t; best_d = d; }
    }
    a[i] = best_d;
    apply(k, req, i, best_d);  // commit (updates sub_/cnt_/egress_/bdcnt_/nunc_/prep_acc_)
  }

  r.exact = false;
  evaluate_armed(k, req, a, &r);
  return r;
}

// ── free-function wrappers (construct a local LoaderSolver) ──
SolveResult solve(const LoaderConstants& k, const SolveRequest& req) {
  LoaderSolver s;
  return s.solve(k, req);
}
SolveResult solve_greedy(const LoaderConstants& k, const SolveRequest& req) {
  LoaderSolver s;
  return s.solve_greedy(k, req);
}
double evaluate(const LoaderConstants& k, const SolveRequest& req,
                const std::vector<int>& assignment, SolveResult* out) {
  LoaderSolver s;
  return s.evaluate(k, req, assignment, out);
}

// The ONLY two instantiations (see the header note). Same TU + same flags as
// the historical non-template class: the 64 instantiation's member bodies are
// token-identical to the pre-template code (NMax is the only substitution), so
// its FP expression order — hence every decision — is unchanged (guarded by
// GpuLoaderSolver.Solver64GoldenRegression).
template class BasicLoaderSolver<kMaxExperts>;
template class BasicLoaderSolver<kMaxExpertsLarge>;

}  // namespace layerstorm::gpu_loader
