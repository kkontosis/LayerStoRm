#pragma once
// Loader evict-score board — EXTERNALLY score-driven, duplicate-aware ranking
// (spec/tickets/EVICTBOARD_EXTERNAL_SCORES.md; supersedes the internal LRU
// recency-clock of spec/tickets/LOADER_STATS_LOCALITY.md §A / Phase 2).
//
// Per-(GPU, resident-expert) eviction-score store the I8 loader uses to build
// the convex evict_cum curve (cheapest victims first). Each GPU owns:
//   - a slot pool (SoA) of its resident experts' raw + effective scores,
//   - an indexed MIN-heap on the EFFECTIVE score (heap top = cheapest victim).
// A cross-GPU index (experts_) maps each expert key to all GPUs holding a copy,
// so a replicated expert's M entries can be RANKED cross-GPU on every touch.
//
// SCORE SEMANTICS (exact):
//   * A raw score is a real number, supplied from OUTSIDE the board. LOWEST =
//     evict first, HIGHEST = keep. The board never invents scores (no internal
//     LRU clock); the external SOURCE is out of scope for this module.
//   * baseScoreValue (default 0.9, configurable) has TWO roles: (a) the default
//     raw score a newly-inserted expert gets (insert_default), and (b) the
//     per-rank trivialization penalty below.
//   * NEVER read the stored raw score directly for ordering — go through
//     score(g, k): an expert resident on M >= 1 GPUs has its M entries RANKED by
//     raw DESC, ties broken by GPU index ASC; the largest gets rank r=0.
//       score(g, k) = max(0, raw - r*baseScoreValue).
//     The primary copy keeps its raw score; secondary duplicates are trivialized
//     toward 0 (-> evicted first, INV-0.4). M==1 => r=0 => effective == raw.
//   * The per-GPU heap is ordered by the EFFECTIVE score, so an entry's heap key
//     depends on the expert's cross-GPU rank. Whenever an expert's entry-set
//     changes (a copy inserted/evicted) OR a copy's raw score changes, ALL its M
//     entries are re-ranked and re-sifted in their respective per-GPU heaps
//     (rerank_expert). M <= num_gpus (small).
//
// SOLE-MUTATOR DISCIPLINE (INV-LOADER-EVICTBOARD-1): raw scores are writable
// ONLY through the 6 API methods (update, insert_default, evict, batch_upsert,
// batch_evict, update_existing_only) + set_base_score; after every call the
// {raw, eff, per-GPU min-heaps, heap_pos inverse, key_to_slot maps, experts_
// index, ranks} are mutually consistent.
//
// HOT PATH: cheapest_score(g) is O(1); cheapest_scores_sorted(g, k, out) and its
// key-returning twin cheapest_keys(g, k, out) extract the k lowest-effective-
// score residents (scores / keys) ASCENDING via a bounded partial heap walk,
// O(k*log k) -- never the GPU's full resident set.
//
// The board is a HINT store, not authoritative residency (INV-LOADER-
// EVICTBOARD-2 -- ExpertCache is truth); a stale out-of-band entry only
// trivializes toward a cheap victim, the safe default.
//
// CUDA-free (layerstorm_core / INV-GPU-1). Owned by the daemon top-layer.
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "core/memory/eviction_policy.h"    // ExpertKey
#include "core/memory/residency_listener.h"  // ResidencyListener (membership channel)

namespace layerstorm::gpu_loader {

// Default raw score for a freshly-inserted expert AND the per-rank trivialization
// penalty (baseScoreValue, both roles -- spec section 0.3). Configurable per-board
// via set_base_score().
inline constexpr double kDefaultBaseScore = 0.9;

// Effective score returned for an empty GPU's "cheapest victim" -- maximally
// evictable (the safe default until a real resident exists).
inline constexpr double kEmptyScore = 0.0;

// Sentinel returned by score()/raw_of() for a non-resident (g, k). Effective
// scores are always >= 0, so a negative sentinel is unambiguous.
inline constexpr double kAbsentScore = -1.0;

// Batch payloads.
struct ScoreUpdate {
  int gpu;
  memory::ExpertKey key;
  double score;
};
struct KeyOnGpu {
  int gpu;
  memory::ExpertKey key;
};

// EvictScoreBoard authoritatively tracks ExpertCache STABLE residency by being
// the cache's memory::ResidencyListener (TD-EVICT-BOARD-DESYNC). The cache fires
// on_resident_added / on_resident_removed at its own choke-points, so the board's
// per-GPU residency CANNOT drift from the cache. The listener overrides drive
// MEMBERSHIP ONLY (insert at the current recency clock / evict); score-feeding
// (decay) stays a separate public API (least privilege).
class EvictScoreBoard : public memory::ResidencyListener {
 public:
  // num_gpus = M; cap_per_gpu = max residents tracked per GPU (a hint; the pool
  // grows on demand if exceeded).
  EvictScoreBoard(int num_gpus, int cap_per_gpu);

  int num_gpus() const { return static_cast<int>(gpus_.size()); }

  // baseScoreValue (spec section 0.3): default raw + per-rank penalty. Setting it
  // re-ranks + re-sifts ALL experts on ALL GPUs (off-hot-path config call).
  void set_base_score(double v);
  double base_score() const { return base_score_; }

  // --- The 6 mutators (the ONLY way raw/eff/heap/experts_ change) ------------

  // 1. single upsert: present(g,k) -> overwrite raw = score; absent -> INSERT a
  //    new entry with raw = score (honors the supplied value, NOT the default).
  //    Re-ranks the touched expert's M copies. (spec section 0.9.)
  void update(int g, memory::ExpertKey k, double score);

  // 2. insert new with the default (baseScoreValue) -- FETCH place trigger.
  //    present(g,k) -> re-set raw = base_score_ (re-touch); absent -> insert.
  void insert_default(int g, memory::ExpertKey k);

  // 3. evict one (expert,gpu) entry -- evict / D2H trigger. No-op if absent.
  //    Frees the slot, drops the ref from experts_[k] (erases k if last), and
  //    re-ranks the SURVIVING copies (a removed duplicate un-trivializes a
  //    survivor on its own GPU).
  void evict(int g, memory::ExpertKey k);

  // 4. batch score upsert (update-or-insert many). Applies all slot mutations
  //    first, then re-ranks each DISTINCT touched expert once (spec T8).
  void batch_upsert(std::span<const ScoreUpdate> ups);

  // 5. batch evict (remove many). Same two-phase pattern.
  void batch_evict(std::span<const KeyOnGpu> evs);

  // 6. batch score UPDATE, NO upsert (update existing only). SILENTLY ignores
  //    non-resident (g,k) -- cannot desync (spec T9).
  void update_existing_only(std::span<const ScoreUpdate> ups);

  // Recency-decay hook for an external time-based scorer: multiply EVERY
  // resident's raw score by `factor` (0<factor<=1), then re-rank + re-sift all
  // experts. Pair with per-use re-touch (update/insert_default back to base) so
  // recently-used experts stay high (KEEP) and untouched ones decay toward 0
  // (EVICT). A uniform scale preserves cross-GPU raw-DESC ranks, but the
  // effective score max(0, raw - r*base) is non-linear in raw, so we re-rank to
  // stay exact (cheap; r=0 in the common no-duplicate case). No-op at factor==1.
  void decay_all(double factor);

  // --- Always-on per-access recency (TD-FAR-EVICT-REROUTE step 3) -----------
  // A per-slot last-touch sequence so the board orders victims by TRUE LRU with
  // NO external decay feed required (within-token ties collapse from ~K×layers to
  // ~K per layer-visit → the per-access-LRU ceiling). The recency clock is the raw
  // score: HIGHER = more recently used = KEEP, LOWER = older = evict first (same
  // polarity as the external scores). Cheap: O(1) stamp; the heap re-sift is the
  // existing cost.
  //
  // advance_recency() bumps the clock once per layer-visit (the caller's
  // per-layer-grouped LRU granularity). on_resident_added stamps the CURRENT clock
  // (a fresh resident is the most-recently-touched), and touch_existing re-stamps
  // an already-resident expert on routed use. All experts touched/added in the
  // same layer-visit share the clock value (per-layer-grouped LRU).
  void advance_recency() { recency_clock_ += 1.0; }
  double recency_now() const { return recency_clock_; }

  // Re-stamp an ALREADY-resident (g,k) to the current recency clock (routed use).
  // SILENTLY ignores a non-resident (g,k) — it must NOT insert (that would re-open
  // desync source (b): a board entry for a not-yet-resident fetch). A freshly
  // fetched miss is instead stamped by on_resident_added when the cache reserves
  // its slot. No-op cost when absent.
  void touch_existing(int g, memory::ExpertKey k);
  // Residency-aware touch (TD-LOADER-REUSE-ENGINE-FIDELITY fix): re-stamp EVERY
  // resident copy of k — wherever it lives — to recency_clock_ + raw_bonus, one
  // rerank. Under ACT a hit can execute on a REPLICA device (reroute), where the
  // target-keyed touch_existing(g,k) was a silent no-op and the copy's recency
  // went stale (evicted young — the −4pp engine-vs-designed-policy gap). The
  // raw_bonus carries the decayed-frequency protection (policy lab: freq d=0.1
  // w≈60 engine-clock units ⇒ ≤ ~8 tokens of extra protection for hot experts).
  // No-op for keys with no resident copy (never inserts — desync source b).
  void touch_existing_all(memory::ExpertKey k, double raw_bonus = 0.0);

  // --- ResidencyListener (the membership channel — TD-EVICT-BOARD-DESYNC) ----
  // The cache fires these at its stable-zone choke-points. They mutate board
  // MEMBERSHIP only: added → insert at the current recency clock; removed → evict.
  void on_resident_added(memory::ExpertKey key, int gpu_idx) override;
  void on_resident_removed(memory::ExpertKey key, int gpu_idx) override;

  // --- Duplicate-aware accessor (the ONLY ordering/eviction read) -----------
  // Effective score = max(0, raw - rank*base_score_). kAbsentScore if not
  // resident. O(1) -- reads the cached eff (kept current by the mutators).
  double score(int g, memory::ExpertKey k) const;

  // --- Hot-path reads --------------------------------------------------------

  // Cheapest victim's EFFECTIVE score on GPU g (min-heap top), or kEmptyScore if
  // empty. O(1).
  double cheapest_score(int g) const;

  // The k LOWEST-effective-score residents on GPU g, ASCENDING (cheapest victim
  // first) -- the prefix the convex evict_cum curve consumes. Bounded partial
  // walk of the per-GPU min-heap, O(k*log k), touching only ~k heap nodes.
  // `out` is resized to the count written; also reports the GPU's total resident
  // count (the curve charges max beyond the gathered tail).
  int cheapest_scores_sorted(int g, int k, std::vector<double>& out,
                             int* resident_count_out = nullptr) const;

  // The k LOWEST-effective-score resident KEYS on GPU g, ASCENDING (cheapest
  // victim first) -- the key-returning twin of cheapest_scores_sorted. Same
  // bounded partial walk of the per-GPU min-heap, O(k*log k); the emitted order
  // matches cheapest_scores_sorted slot-for-slot (same total order: eff
  // ascending, tie-break by slot index). `out` is resized to the count written;
  // also reports the GPU's total resident count.
  // The board AUTHORITATIVELY tracks ExpertCache STABLE residency (it is the
  // cache's ResidencyListener — TD-EVICT-BOARD-DESYNC), so these keys ARE resident
  // in the cache's stable zone: the reroute eviction fallback drives victim
  // selection directly off them (the O(k·log N) heap walk). A debug-build parity
  // assert (board residency == cache stable residency) guards any future bypass.
  int cheapest_keys(int g, int k, std::vector<memory::ExpertKey>& out,
                    int* resident_count_out = nullptr) const;

  // --- Introspection (tests / accounting) ------------------------------------
  int resident_count(int g) const;
  bool is_resident(int g, memory::ExpertKey k) const;
  // Enumerate ALL resident keys on GPU g (unordered). For the debug parity assert
  // and tests; not a hot-path read.
  void resident_keys(int g, std::vector<memory::ExpertKey>& out) const;
  double raw_of(int g, memory::ExpertKey k) const;  // kAbsentScore if absent
  int rank_of(int g, memory::ExpertKey k) const;    // cross-GPU rank, -1 if absent
  int dup_count(memory::ExpertKey k) const;         // M for this expert
  // Validate the sync invariant (INV-LOADER-EVICTBOARD-1) for GPU g. Test-only.
  bool check_invariant(int g) const;

 private:
  struct PerGpu {
    std::vector<memory::ExpertKey> key;   // [cap] resident expert per slot
    std::vector<double> raw;              // [cap] externally-supplied raw score
    std::vector<double> eff;             // [cap] cached effective score
    std::vector<int> heap_pos;            // [cap] slot's index in heap (-1 = none)
    std::vector<int> free_slots;          // recycled slot indices
    std::unordered_map<memory::ExpertKey, int> key_to_slot;
    std::vector<int> heap;                // min-heap of slot indices, keyed eff[]
  };

  // Cross-GPU index entry: a copy of an expert lives at (gpu, slot).
  struct EntryRef {
    int gpu;
    int slot;
  };

  // Slot pool ops.
  int alloc_slot(PerGpu& pg, memory::ExpertKey k, double raw_score);
  void free_slot(PerGpu& pg, int slot);

  // Indexed min-heap ops on slot indices, ordered by pg.eff[slot] ascending.
  bool heap_less(const PerGpu& pg, int slot_a, int slot_b) const;
  void heap_swap(PerGpu& pg, int hi, int hj);
  void sift_up(PerGpu& pg, int hi);
  void sift_down(PerGpu& pg, int hi);
  void heap_push(PerGpu& pg, int slot);
  void heap_remove(PerGpu& pg, int slot);
  void heap_update(PerGpu& pg, int slot);  // after eff[slot] changed

  // experts_ index ops.
  void index_add(memory::ExpertKey k, int gpu, int slot);
  void index_remove(memory::ExpertKey k, int gpu);

  // THE CASCADE: re-rank all M copies of expert k (raw DESC, gpu ASC), recompute
  // each eff = max(0, raw - r*base), write eff[], and re-sift each per-GPU heap.
  void rerank_expert(memory::ExpertKey k);

  // Internal single-entry upsert shared by update / insert_default (caller
  // supplies the raw value). Does the slot+index mutation but NOT the re-rank,
  // so batch callers can defer the re-rank to a deduped pass.
  void upsert_raw_no_rerank(int g, memory::ExpertKey k, double raw_score);
  // Internal single-entry evict without the re-rank (batch defers).
  void evict_no_rerank(int g, memory::ExpertKey k);

  std::vector<PerGpu> gpus_;
  std::unordered_map<memory::ExpertKey, std::vector<EntryRef>> experts_;
  int cap_per_gpu_;
  double base_score_ = kDefaultBaseScore;
  // Always-on recency clock (TD-FAR-EVICT-REROUTE step 3): the raw score a fresh
  // add / routed touch is stamped with. Monotonically advanced per layer-visit by
  // advance_recency(). double (not integer) keeps it uniform with the external
  // score path (decay_all multiplies raws); 100 tok × ~29 MoE layers ≈ 2900, no
  // overflow concern.
  double recency_clock_ = 0.0;
};

}  // namespace layerstorm::gpu_loader
