#include "core/gpu_loader/loader_evict_scores.h"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <utility>
#include <vector>

namespace layerstorm::gpu_loader {

EvictScoreBoard::EvictScoreBoard(int num_gpus, int cap_per_gpu)
    : cap_per_gpu_(cap_per_gpu < 0 ? 0 : cap_per_gpu) {
  if (num_gpus < 0) num_gpus = 0;
  gpus_.resize(static_cast<size_t>(num_gpus));
  for (auto& pg : gpus_) {
    pg.key.reserve(cap_per_gpu_);
    pg.raw.reserve(cap_per_gpu_);
    pg.eff.reserve(cap_per_gpu_);
    pg.heap_pos.reserve(cap_per_gpu_);
    pg.heap.reserve(cap_per_gpu_);
  }
}

// --- slot pool ----------------------------------------------------------------

int EvictScoreBoard::alloc_slot(PerGpu& pg, memory::ExpertKey k,
                                double raw_score) {
  int slot;
  if (!pg.free_slots.empty()) {
    slot = pg.free_slots.back();
    pg.free_slots.pop_back();
    pg.key[slot] = k;
    pg.raw[slot] = raw_score;
    pg.eff[slot] = raw_score;  // provisional; rerank_expert recomputes
    pg.heap_pos[slot] = -1;
  } else {
    slot = static_cast<int>(pg.key.size());
    pg.key.push_back(k);
    pg.raw.push_back(raw_score);
    pg.eff.push_back(raw_score);
    pg.heap_pos.push_back(-1);
  }
  pg.key_to_slot[k] = slot;
  return slot;
}

void EvictScoreBoard::free_slot(PerGpu& pg, int slot) {
  heap_remove(pg, slot);
  pg.key_to_slot.erase(pg.key[slot]);
  pg.free_slots.push_back(slot);
}

// --- indexed min-heap on eff[] (lowest effective score = cheapest victim) -----

bool EvictScoreBoard::heap_less(const PerGpu& pg, int sa, int sb) const {
  // min-heap on the EFFECTIVE score: lower eff = cheaper victim = closer to top.
  if (pg.eff[sa] != pg.eff[sb]) return pg.eff[sa] < pg.eff[sb];
  return sa < sb;  // deterministic tie-break by slot index
}

void EvictScoreBoard::heap_swap(PerGpu& pg, int hi, int hj) {
  std::swap(pg.heap[hi], pg.heap[hj]);
  pg.heap_pos[pg.heap[hi]] = hi;
  pg.heap_pos[pg.heap[hj]] = hj;
}

void EvictScoreBoard::sift_up(PerGpu& pg, int hi) {
  while (hi > 0) {
    const int parent = (hi - 1) / 2;
    if (heap_less(pg, pg.heap[hi], pg.heap[parent])) {
      heap_swap(pg, hi, parent);
      hi = parent;
    } else {
      break;
    }
  }
}

void EvictScoreBoard::sift_down(PerGpu& pg, int hi) {
  const int n = static_cast<int>(pg.heap.size());
  while (true) {
    const int l = 2 * hi + 1, r = 2 * hi + 2;
    int smallest = hi;
    if (l < n && heap_less(pg, pg.heap[l], pg.heap[smallest])) smallest = l;
    if (r < n && heap_less(pg, pg.heap[r], pg.heap[smallest])) smallest = r;
    if (smallest == hi) break;
    heap_swap(pg, hi, smallest);
    hi = smallest;
  }
}

void EvictScoreBoard::heap_push(PerGpu& pg, int slot) {
  pg.heap_pos[slot] = static_cast<int>(pg.heap.size());
  pg.heap.push_back(slot);
  sift_up(pg, pg.heap_pos[slot]);
}

void EvictScoreBoard::heap_remove(PerGpu& pg, int slot) {
  const int hi = pg.heap_pos[slot];
  if (hi < 0) return;  // not in heap
  const int last = static_cast<int>(pg.heap.size()) - 1;
  if (hi != last) {
    pg.heap[hi] = pg.heap[last];
    pg.heap_pos[pg.heap[hi]] = hi;
  }
  pg.heap.pop_back();
  pg.heap_pos[slot] = -1;
  if (hi <= last - 1 && hi < static_cast<int>(pg.heap.size())) {
    // restore heap order at hi (the moved element may go up or down).
    sift_up(pg, hi);
    sift_down(pg, hi);
  }
}

void EvictScoreBoard::heap_update(PerGpu& pg, int slot) {
  const int hi = pg.heap_pos[slot];
  if (hi < 0) {  // not yet in heap -> insert
    heap_push(pg, slot);
    return;
  }
  sift_up(pg, hi);
  sift_down(pg, hi);
}

// --- experts_ cross-GPU index -------------------------------------------------

void EvictScoreBoard::index_add(memory::ExpertKey k, int gpu, int slot) {
  auto& refs = experts_[k];
  for (auto& r : refs) {
    if (r.gpu == gpu) {  // already present on this GPU -> refresh slot
      r.slot = slot;
      return;
    }
  }
  refs.push_back(EntryRef{gpu, slot});
}

void EvictScoreBoard::index_remove(memory::ExpertKey k, int gpu) {
  auto it = experts_.find(k);
  if (it == experts_.end()) return;
  auto& refs = it->second;
  for (size_t i = 0; i < refs.size(); ++i) {
    if (refs[i].gpu == gpu) {
      refs[i] = refs.back();
      refs.pop_back();
      break;
    }
  }
  if (refs.empty()) experts_.erase(it);
}

// --- THE CASCADE: rank all M copies of an expert, recompute eff, re-sift ------

void EvictScoreBoard::rerank_expert(memory::ExpertKey k) {
  auto it = experts_.find(k);
  if (it == experts_.end()) return;  // no copies left -> nothing to rank
  auto& refs = it->second;
  const int m = static_cast<int>(refs.size());
  // Order copies by (raw DESC, gpu ASC); the index in this order is the rank r.
  // m <= num_gpus (small) -> a tiny local index array + std::sort is fine.
  std::array<int, 64> order_inline{};
  std::vector<int> order_vec;
  int* order = order_inline.data();
  if (m > static_cast<int>(order_inline.size())) {
    order_vec.resize(static_cast<size_t>(m));
    order = order_vec.data();
  }
  for (int i = 0; i < m; ++i) order[i] = i;
  std::sort(order, order + m, [&](int a, int b) {
    const double ra = gpus_[refs[a].gpu].raw[refs[a].slot];
    const double rb = gpus_[refs[b].gpu].raw[refs[b].slot];
    if (ra != rb) return ra > rb;            // raw DESC
    return refs[a].gpu < refs[b].gpu;        // tie: gpu index ASC
  });
  for (int r = 0; r < m; ++r) {
    const EntryRef e = refs[order[r]];
    PerGpu& pg = gpus_[e.gpu];
    double new_eff = pg.raw[e.slot] - static_cast<double>(r) * base_score_;
    if (new_eff < 0.0) new_eff = 0.0;        // max(0, ...)
    if (pg.eff[e.slot] != new_eff) {
      pg.eff[e.slot] = new_eff;
      heap_update(pg, e.slot);               // re-sift (eff changed)
    } else {
      // eff unchanged but the slot may be newly allocated and not yet in the
      // heap (heap_pos == -1); heap_update inserts it in that case.
      if (pg.heap_pos[e.slot] < 0) heap_update(pg, e.slot);
    }
  }
}

// --- internal single-entry mutations (no re-rank; batch defers) ---------------

void EvictScoreBoard::upsert_raw_no_rerank(int g, memory::ExpertKey k,
                                           double raw_score) {
  if (g < 0 || g >= num_gpus()) return;
  PerGpu& pg = gpus_[g];
  auto it = pg.key_to_slot.find(k);
  if (it != pg.key_to_slot.end()) {
    pg.raw[it->second] = raw_score;          // overwrite (re-rank fixes eff)
  } else {
    const int slot = alloc_slot(pg, k, raw_score);
    index_add(k, g, slot);                   // heap insert happens in rerank
  }
}

void EvictScoreBoard::evict_no_rerank(int g, memory::ExpertKey k) {
  if (g < 0 || g >= num_gpus()) return;
  PerGpu& pg = gpus_[g];
  auto it = pg.key_to_slot.find(k);
  if (it == pg.key_to_slot.end()) return;    // not resident -> no-op
  free_slot(pg, it->second);
  index_remove(k, g);
}

// --- the 6 public mutators ----------------------------------------------------

void EvictScoreBoard::update(int g, memory::ExpertKey k, double score) {
  if (g < 0 || g >= num_gpus()) return;
  upsert_raw_no_rerank(g, k, score);
  rerank_expert(k);
}

void EvictScoreBoard::insert_default(int g, memory::ExpertKey k) {
  if (g < 0 || g >= num_gpus()) return;
  upsert_raw_no_rerank(g, k, base_score_);
  rerank_expert(k);
}

void EvictScoreBoard::evict(int g, memory::ExpertKey k) {
  if (g < 0 || g >= num_gpus()) return;
  PerGpu& pg = gpus_[g];
  if (pg.key_to_slot.find(k) == pg.key_to_slot.end()) return;  // no-op
  evict_no_rerank(g, k);
  rerank_expert(k);  // survivors (if any) un-trivialize; erased key -> no-op
}

void EvictScoreBoard::batch_upsert(std::span<const ScoreUpdate> ups) {
  // Phase 1: apply all slot/index mutations, collect the distinct touched keys.
  std::unordered_set<memory::ExpertKey> touched;
  touched.reserve(ups.size());
  for (const auto& u : ups) {
    if (u.gpu < 0 || u.gpu >= num_gpus()) continue;
    upsert_raw_no_rerank(u.gpu, u.key, u.score);
    touched.insert(u.key);
  }
  // Phase 2: re-rank each distinct touched expert ONCE (final entry-set).
  for (const auto& k : touched) rerank_expert(k);
}

void EvictScoreBoard::batch_evict(std::span<const KeyOnGpu> evs) {
  std::unordered_set<memory::ExpertKey> touched;
  touched.reserve(evs.size());
  for (const auto& e : evs) {
    if (e.gpu < 0 || e.gpu >= num_gpus()) continue;
    PerGpu& pg = gpus_[e.gpu];
    if (pg.key_to_slot.find(e.key) == pg.key_to_slot.end()) continue;  // no-op
    evict_no_rerank(e.gpu, e.key);
    touched.insert(e.key);
  }
  for (const auto& k : touched) rerank_expert(k);  // survivors un-trivialize
}

void EvictScoreBoard::update_existing_only(std::span<const ScoreUpdate> ups) {
  std::unordered_set<memory::ExpertKey> touched;
  touched.reserve(ups.size());
  for (const auto& u : ups) {
    if (u.gpu < 0 || u.gpu >= num_gpus()) continue;
    PerGpu& pg = gpus_[u.gpu];
    auto it = pg.key_to_slot.find(u.key);
    if (it == pg.key_to_slot.end()) continue;  // SILENTLY ignore absent (T9)
    pg.raw[it->second] = u.score;
    touched.insert(u.key);
  }
  for (const auto& k : touched) rerank_expert(k);
}

// --- always-on recency + ResidencyListener membership channel -----------------

void EvictScoreBoard::touch_existing(int g, memory::ExpertKey k) {
  if (g < 0 || g >= num_gpus()) return;
  PerGpu& pg = gpus_[g];
  auto it = pg.key_to_slot.find(k);
  if (it == pg.key_to_slot.end()) return;  // MUST NOT insert (no desync source b)
  pg.raw[it->second] = recency_clock_;     // re-stamp to current layer recency
  rerank_expert(k);
}

void EvictScoreBoard::touch_existing_all(memory::ExpertKey k, double raw_bonus) {
  auto it = experts_.find(k);
  if (it == experts_.end()) return;  // not resident anywhere — never insert
  const double raw = recency_clock_ + raw_bonus;
  auto& refs = it->second;
  if (refs.size() == 1) {
    // Hot-path fast case (no duplicates — the overwhelmingly common decode
    // shape): rank 0 ⇒ eff = max(0, raw); one store + one heap re-sift.
    // Byte-identical to rerank_expert's m==1 outcome, minus its second
    // experts_ hash find and the sort/array scaffolding.
    const EntryRef r = refs[0];
    PerGpu& pg = gpus_[r.gpu];
    pg.raw[r.slot] = raw;
    const double new_eff = raw < 0.0 ? 0.0 : raw;
    if (pg.eff[r.slot] != new_eff) {
      pg.eff[r.slot] = new_eff;
      heap_update(pg, r.slot);
    } else if (pg.heap_pos[r.slot] < 0) {
      heap_update(pg, r.slot);
    }
    return;
  }
  for (const EntryRef& r : refs) {
    PerGpu& pg = gpus_[r.gpu];
    pg.raw[r.slot] = raw;
  }
  rerank_expert(k);
}

void EvictScoreBoard::on_resident_added(memory::ExpertKey key, int gpu_idx) {
  // Membership: a fresh stable resident is the most-recently-touched → stamp the
  // CURRENT recency clock (NOT base — that is the external decay default). Upsert:
  // overwrite if a stale entry somehow remains.
  if (gpu_idx < 0 || gpu_idx >= num_gpus()) return;
  upsert_raw_no_rerank(gpu_idx, key, recency_clock_);
  rerank_expert(key);
}

void EvictScoreBoard::on_resident_removed(memory::ExpertKey key, int gpu_idx) {
  evict(gpu_idx, key);  // membership-only removal (no-op if absent)
}

void EvictScoreBoard::resident_keys(int g,
                                    std::vector<memory::ExpertKey>& out) const {
  out.clear();
  if (g < 0 || g >= num_gpus()) return;
  const PerGpu& pg = gpus_[g];
  out.reserve(pg.key_to_slot.size());
  for (const auto& [k, slot] : pg.key_to_slot) out.push_back(k);
}

void EvictScoreBoard::set_base_score(double v) {
  base_score_ = v;
  // Global re-rank: every expert's effective scores depend on base_score_.
  // Snapshot the keys first (rerank_expert does not mutate experts_'s key set,
  // but iterate a copy to be safe against rehashes).
  std::vector<memory::ExpertKey> keys;
  keys.reserve(experts_.size());
  for (const auto& [k, refs] : experts_) keys.push_back(k);
  for (const auto& k : keys) rerank_expert(k);
}

void EvictScoreBoard::decay_all(double factor) {
  if (factor == 1.0) return;
  for (auto& pg : gpus_)
    for (const auto& [k, slot] : pg.key_to_slot) pg.raw[slot] *= factor;
  // eff = max(0, raw - r*base) is non-linear in raw → re-rank every expert
  // (also re-sifts each per-GPU heap). Snapshot keys (guard against rehash).
  std::vector<memory::ExpertKey> keys;
  keys.reserve(experts_.size());
  for (const auto& [k, refs] : experts_) keys.push_back(k);
  for (const auto& k : keys) rerank_expert(k);
}

// --- duplicate-aware accessor + hot-path reads --------------------------------

double EvictScoreBoard::score(int g, memory::ExpertKey k) const {
  if (g < 0 || g >= num_gpus()) return kAbsentScore;
  const PerGpu& pg = gpus_[g];
  auto it = pg.key_to_slot.find(k);
  if (it == pg.key_to_slot.end()) return kAbsentScore;
  return pg.eff[it->second];
}

double EvictScoreBoard::cheapest_score(int g) const {
  if (g < 0 || g >= num_gpus()) return kEmptyScore;
  const PerGpu& pg = gpus_[g];
  if (pg.heap.empty()) return kEmptyScore;
  return pg.eff[pg.heap[0]];
}

namespace {
// Shared frontier partial-walk for the two k-cheapest accessors
// (cheapest_scores_sorted -> eff, cheapest_keys -> key). Invokes emit(slot) for
// the `want` lowest-eff resident SLOT indices of `heap` ASCENDING (cheapest
// victim first), WITHOUT mutating `heap`: a frontier min-heap of heap NODE
// positions ordered by the underlying slot's eff (tie-break by slot index, the
// same total order as the per-GPU heap). Pop the frontier `want` times, pushing
// each popped node's two children. Touches O(want*log want) nodes -- never the
// full resident set ("k smallest from a binary heap" in O(k log k)). Returns
// the number of slots emitted. `want` must already be clamped to <= heap size.
template <typename Emit>
int walk_cheapest_slots(const std::vector<int>& heap,
                        const std::vector<double>& eff, int want, Emit&& emit) {
  const int total = static_cast<int>(heap.size());
  if (want <= 0 || total == 0) return 0;
  if (want > total) want = total;
  // Frontier storage: small inline buffer for the common bounded walks (curve
  // needs k<=N routed, victim gather k~2K+16) — no heap allocation on the
  // per-layer hot path; the rare full-residency widen falls back to a vector.
  const int fr_cap = want * 2 + 2;
  std::array<int, 2 * 64 + 2> fr_inline;  // want <= kMaxExperts covers inline
  std::vector<int> frontier_vec;
  int* fr;
  if (fr_cap <= static_cast<int>(fr_inline.size())) {
    fr = fr_inline.data();
  } else {
    frontier_vec.assign(static_cast<size_t>(fr_cap), 0);
    fr = frontier_vec.data();
  }
  int fr_n = 0;
  auto fr_less = [&](int pa, int pb) {  // compare by underlying slot eff
    const int sa = heap[pa], sb = heap[pb];
    if (eff[sa] != eff[sb]) return eff[sa] < eff[sb];
    return sa < sb;
  };
  auto fr_sift_up = [&](int i) {
    while (i > 0) {
      const int par = (i - 1) / 2;
      if (fr_less(fr[i], fr[par])) { std::swap(fr[i], fr[par]); i = par; }
      else break;
    }
  };
  auto fr_sift_down = [&](int i) {
    while (true) {
      const int l = 2 * i + 1, r = 2 * i + 2;
      int s = i;
      if (l < fr_n && fr_less(fr[l], fr[s])) s = l;
      if (r < fr_n && fr_less(fr[r], fr[s])) s = r;
      if (s == i) break;
      std::swap(fr[i], fr[s]); i = s;
    }
  };
  auto fr_push = [&](int pos) {
    if (fr_n >= fr_cap) return;  // bounded; defensive
    fr[fr_n++] = pos; fr_sift_up(fr_n - 1);
  };
  fr_push(0);  // root of the underlying heap = global min
  int got = 0;
  for (; got < want && fr_n > 0; ++got) {
    const int pos = fr[0];                 // frontier min = next smallest
    fr[0] = fr[--fr_n]; if (fr_n > 0) fr_sift_down(0);
    emit(heap[pos]);
    const int l = 2 * pos + 1, r = 2 * pos + 2;
    if (l < total) fr_push(l);
    if (r < total) fr_push(r);
  }
  return got;  // already ascending (cheapest first)
}
}  // namespace

int EvictScoreBoard::cheapest_scores_sorted(int g, int k,
                                            std::vector<double>& out,
                                            int* resident_count_out) const {
  out.clear();
  if (resident_count_out) *resident_count_out = 0;
  if (g < 0 || g >= num_gpus() || k <= 0) return 0;
  const PerGpu& pg = gpus_[g];
  const int total = static_cast<int>(pg.heap.size());
  if (resident_count_out) *resident_count_out = total;
  if (total == 0) return 0;
  const int want = std::min(k, total);
  out.reserve(want);
  return walk_cheapest_slots(pg.heap, pg.eff, want,
                             [&](int slot) { out.push_back(pg.eff[slot]); });
}

int EvictScoreBoard::cheapest_keys(int g, int k,
                                   std::vector<memory::ExpertKey>& out,
                                   int* resident_count_out) const {
  out.clear();
  if (resident_count_out) *resident_count_out = 0;
  if (g < 0 || g >= num_gpus() || k <= 0) return 0;
  const PerGpu& pg = gpus_[g];
  const int total = static_cast<int>(pg.heap.size());
  if (resident_count_out) *resident_count_out = total;
  if (total == 0) return 0;
  const int want = std::min(k, total);
  out.reserve(want);
  return walk_cheapest_slots(pg.heap, pg.eff, want,
                             [&](int slot) { out.push_back(pg.key[slot]); });
}

// --- introspection ------------------------------------------------------------

int EvictScoreBoard::resident_count(int g) const {
  if (g < 0 || g >= num_gpus()) return 0;
  return static_cast<int>(gpus_[g].key_to_slot.size());
}

bool EvictScoreBoard::is_resident(int g, memory::ExpertKey k) const {
  if (g < 0 || g >= num_gpus()) return false;
  return gpus_[g].key_to_slot.count(k) > 0;
}

double EvictScoreBoard::raw_of(int g, memory::ExpertKey k) const {
  if (g < 0 || g >= num_gpus()) return kAbsentScore;
  const PerGpu& pg = gpus_[g];
  auto it = pg.key_to_slot.find(k);
  if (it == pg.key_to_slot.end()) return kAbsentScore;
  return pg.raw[it->second];
}

int EvictScoreBoard::rank_of(int g, memory::ExpertKey k) const {
  if (g < 0 || g >= num_gpus()) return -1;
  if (gpus_[g].key_to_slot.find(k) == gpus_[g].key_to_slot.end()) return -1;
  auto it = experts_.find(k);
  if (it == experts_.end()) return -1;
  const auto& refs = it->second;
  // rank = # of copies strictly "ahead" of this one under (raw DESC, gpu ASC).
  const double my_raw = raw_of(g, k);
  int rank = 0;
  for (const auto& e : refs) {
    if (e.gpu == g) continue;
    const double er = gpus_[e.gpu].raw[e.slot];
    if (er > my_raw || (er == my_raw && e.gpu < g)) ++rank;
  }
  return rank;
}

int EvictScoreBoard::dup_count(memory::ExpertKey k) const {
  auto it = experts_.find(k);
  return it == experts_.end() ? 0 : static_cast<int>(it->second.size());
}

bool EvictScoreBoard::check_invariant(int g) const {
  if (g < 0 || g >= num_gpus()) return false;
  const PerGpu& pg = gpus_[g];
  // heap size == resident count; heap is a permutation of occupied slots.
  if (pg.heap.size() != pg.key_to_slot.size()) return false;
  for (int hi = 0; hi < static_cast<int>(pg.heap.size()); ++hi) {
    const int slot = pg.heap[hi];
    if (pg.heap_pos[slot] != hi) return false;  // inverse consistent
    auto it = pg.key_to_slot.find(pg.key[slot]);
    if (it == pg.key_to_slot.end() || it->second != slot) return false;
    // eff must equal max(0, raw - rank*base) for this slot's expert.
    const int r = rank_of(g, pg.key[slot]);
    double want_eff = pg.raw[slot] - static_cast<double>(r) * base_score_;
    if (want_eff < 0.0) want_eff = 0.0;
    if (pg.eff[slot] != want_eff) return false;
    // experts_ index must list this (gpu, slot).
    auto eit = experts_.find(pg.key[slot]);
    if (eit == experts_.end()) return false;
    bool found = false;
    for (const auto& e : eit->second)
      if (e.gpu == g && e.slot == slot) { found = true; break; }
    if (!found) return false;
    // min-heap order vs children (on eff).
    const int l = 2 * hi + 1, r2 = 2 * hi + 2;
    if (l < static_cast<int>(pg.heap.size()) && heap_less(pg, pg.heap[l], slot))
      return false;
    if (r2 < static_cast<int>(pg.heap.size()) && heap_less(pg, pg.heap[r2], slot))
      return false;
  }
  // every occupied slot is in the heap.
  for (const auto& [k, slot] : pg.key_to_slot)
    if (pg.heap_pos[slot] < 0) return false;
  return true;
}

}  // namespace layerstorm::gpu_loader
