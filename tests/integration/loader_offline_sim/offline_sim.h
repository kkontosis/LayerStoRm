#pragma once
// CPU-only, GPU-free offline replay of the I8 GPU-loader eviction decisions.
//
// UPGRADE (fab1-offline-sim-engine): the sim now drives the REAL engine eviction
// — a CPU-constructible memory::ExpertCache (via NullBackends, 460 stable slots/
// GPU) with the gpu_loader::EvictScoreBoard registered as its ResidencyListener,
// and the SHARED gpu_loader::apply_far_evictions victim-selection helper (the
// SAME function the daemon's handle_fetch_and_run_moe calls). Placement (ACT
// reroute) goes through the REAL gpu_loader::LoaderSolver; the per-token time is
// the REAL T(j) cost model (LoaderSolver::evaluate). Because residency, scoring,
// and eviction all come from the engine classes, the sim hit-rate == the engine
// hit-rate BY CONSTRUCTION (single source of truth — nothing re-implemented).
//
// FIDELITY FIX (fab1-offline-sim-fidelity): it replays the 5 keeper A/B variants
// CPU-only on the SINGLE baseline routed trace (routed_trace.csv), varying ONLY
// the algorithm (ALGO-NOT-DATA). FP routing-drift is treated as negligible; the
// per-policy oracle traces are NOT used to drive any variant (they manufactured a
// spurious decay inversion via wildly different intrinsic cacheability):
//   1. baseline  (e%tp, loader off; orchestrator clean-LRU eviction map)
//   2. plain-ACT NEW    (reroute + always-on per-layer recency, NO decay; board)
//   3. plain-ACT legacy (kill-switch LS_EVICT_LRU_FALLBACK=0 → hash-order)
//   4. ACT+decay NEW    (recency + LS_EVICT_DECAY ×0.98; board)
//   5. ACT+decay legacy (kill-switch hash-order)
//
// DRIFT-PROBE (fab1-evict-drift-probe): the prior "LS_EVICT_DECAY is a NO-OP"
// claim is refined. decay_all is order-preserving on the per-GPU victim HEAP for
// non-duplicates (eff==raw, uniform scale), so it never directly flips a non-dup
// cheapest_keys victim; BUT it scales the raw recency-clock MAGNITUDES that feed
// cheapest_scores_sorted → evict_cum → the placement solver, which DOES flip
// reroute decisions (a real, measurable decision-stream divergence even on a
// fixed trace). The DecisionLog instrumentation + the bounded-recency prototype
// (recency_inc) below quantify both effects. See RESULTS.md.
//
// The oracle CSVs' (oj,j,cached0,cached1) columns are read ONLY for ASSERTION/
// DEBUG, never to drive a sim decision (HARD RULE). A local clean-LRU CacheSim is
// the CEILING ref.
#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/gpu_loader/loader_constants.h"
#include "core/gpu_loader/loader_solver.h"

namespace layerstorm::offline_sim {

// ── recorded routed-expert trace ────────────────────────────────────────────
struct TraceRow {
  int token = 0;
  int layer = 0;
  int slot = 0;        // routed rank within the layer [0,K)
  int expert_idx = 0;
  int bank = 0;        // expert host NUMA source node
};

// One MoE layer instance: the K routed experts of (token, layer), in slot order.
struct LayerInstance {
  int token = 0;
  int layer = 0;
  std::vector<int> expert_idx;  // [K]
  std::vector<int> bank;        // [K] NUMA node per expert
};

// load_trace reads the first 5 columns (token,layer,slot,expert_idx,bank), so it
// works on BOTH routed_trace.csv and the 9-column oracle_*.csv (extra columns are
// ignored). The expert_idx/bank columns are the INPUT workload, not an oracle
// decision — using them to drive the replay is legitimate (algo-not-data rule).
std::vector<TraceRow>      load_trace(const std::string& csv_path);
std::vector<LayerInstance> group_layers(const std::vector<TraceRow>& rows);
int                        num_tokens(const std::vector<TraceRow>& rows);

// ── local clean-LRU per-GPU cache (CEILING reference only) ───────────────────
// Faithful per-GPU LRU residency set keyed by (layer, expert_idx); the prior
// sim's diagnostic. Used ONLY to report the clean-LRU hit-rate ceiling; the
// per-variant PRIMARY metric drives the real ExpertCache instead.
class CacheSim {
 public:
  CacheSim(int num_gpus, int capacity);
  static uint64_t make_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
  }
  bool resident(int gpu, int layer, int expert) const {
    return g_[gpu].map.count(make_key(layer, expert)) != 0;
  }
  int  count(int gpu) const { return static_cast<int>(g_[gpu].order.size()); }
  int  free_slots(int gpu) const { return capacity_ - count(gpu); }
  void touch(int gpu, uint64_t key);
  std::vector<uint64_t> insert(int gpu, uint64_t key,
                               const std::unordered_set<uint64_t>& needed);
 private:
  struct Gpu {
    std::list<uint64_t> order;  // front = LRU, back = MRU
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> map;
  };
  std::vector<Gpu> g_;
  int capacity_;
};

// ── oracle (ground truth — DEBUG + ASSERTION ONLY, NEVER a decision input) ────
struct OracleRow { int oj = 0, j = 0, c0 = 0, c1 = 0; };
struct OracleKey {
  int token = 0, layer = 0, expert = 0;
  bool operator==(const OracleKey& o) const {
    return token == o.token && layer == o.layer && expert == o.expert;
  }
};
struct OracleKeyHash {
  size_t operator()(const OracleKey& k) const {
    return (static_cast<size_t>(k.token) << 40) ^
           (static_cast<size_t>(k.layer) << 16) ^ static_cast<size_t>(k.expert);
  }
};
using OracleMap = std::unordered_map<OracleKey, OracleRow, OracleKeyHash>;
OracleMap load_oracle(const std::string& csv_path);

// ── variant specification + result ──────────────────────────────────────────
struct VariantSpec {
  std::string name;
  std::string workload_csv;   // oracle file supplying the routed trace (cols 1-5)
  bool loader_act = false;    // ACT reroute on (P2)
  bool decay = false;         // LS_EVICT_DECAY recency aging on
  bool lru_fallback = true;   // board cheapest_keys (true) vs legacy hash-order
  bool use_board = true;      // board live (ACT path); false = loader-off baseline
  // DRIFT-PROBE bounded-recency prototype (TD-EVICT-RECENCY-MAGNITUDE). When > 0,
  // the sim STOPS using the board's unbounded internal recency clock (advance_
  // recency / touch_existing, which grows to ~5800 and swamps base_score=0.9) and
  // instead feeds a BOUNDED external recency score through update_existing_only:
  // a sim-side counter incremented by recency_inc per layer-visit, so the spread
  // stays O(base_score) and the rank·base duplicate penalty actually bites. Pure
  // public-API use — ZERO production change. 0.0 = legacy unbounded board clock.
  double recency_inc = 0.0;
};

struct VariantResult {
  std::string name;
  uint64_t lookups = 0;        // routed experts examined (Σ K per MoE layer)
  uint64_t hits = 0;           // resident at the EXECUTED target gpu (no H2D)
  uint64_t transfers = 0;      // H2D transfers = MISSES (keeper definition)
  uint64_t reserve_failures = 0; // non-resident misses that could not reserve
  uint64_t dup_inserts = 0;    // experts that became resident on >1 GPU
  // ── duplicate / eviction instrumentation (fidelity investigation) ──────────
  uint64_t dup_resident_layersum = 0;  // Σ over layer-visits of #keys on >1 GPU
  uint64_t resident_layersum = 0;      // Σ over layer-visits of total residents
  uint64_t reroute_count = 0;          // misses whose target != home (rerouted)
  uint64_t reroute_to_cached = 0;      // reroutes landing on an already-resident GPU (hit, no H2D)
  uint64_t ev_honored = 0;             // far-evict: orchestrator victim honored
  uint64_t ev_rejected = 0;            // far-evict: orchestrator victim refused
  uint64_t ev_fallback = 0;            // far-evict: board/hash local fallback eviction
  // ── EP-combine compute-owner dedup (mirrors dispatch_moe force-ON dedup) ─────
  uint64_t ep_combine_dups = 0;  // Σ over layer-visits of routed experts resident
                                 // on >1 GPU at compute time (the engine's pre-
                                 // dedup double-count SITES; dedup_ep_residency
                                 // collapses each to one owner before the combine)
  uint64_t ep_owner_shifts = 0;  // routed slots whose deduped compute owner !=
                                 // the naive home/reroute target (cost-model
                                 // single-count corrections vs the old exec=target)
  double   max_recency = 0.0;          // final board recency clock value
  int      tokens = 0;
  double   hit_rate = 0.0;          // 1 - transfers/lookups (keeper definition)
  double   transfers_per_token = 0.0;
  double   predicted_us_total = 0.0; // ΣT(j) over all layer-instances
  double   t_pred_ms_per_token = 0.0;// predicted_us_total / tokens / 1000
  double   tok_s_sim = 0.0;          // filled after the affine calibration
  // Oracle comparison (assertion/debug only).
  uint64_t oracle_rows = 0;        // matched (token,layer,expert) rows
  uint64_t cached_divergences = 0; // sim (c0,c1) != oracle (c0,c1)
  std::string first_divergence;
};

// ── DRIFT-PROBE decision-stream instrumentation (sim-local; NOT production) ───
// Captured per run when a non-null DecisionLog is passed to run_variant; lets the
// drift-probe test diff the decay-on vs decay-off decision streams entry-for-
// entry on the SINGLE fixed trace. Records every routed-expert lookup (its post-
// reroute target + hit/miss) and every eviction the run performs (the victim
// key). Used ONLY for analysis; never feeds back into any decision.
struct DecisionLog {
  struct Lookup {
    int token, layer, expert, home, target;
    uint8_t was_cached;  // resident at e%tp home at solve time
    uint8_t hit;         // counted a hit (no H2D)
  };
  struct Evict {
    int token, layer, gpu;
    uint32_t v_layer;    // evicted victim's (layer,expert)
    uint16_t v_expert;
  };
  std::vector<Lookup> lookups;
  std::vector<Evict>  evicts;
};

// Replay one variant through the REAL ExpertCache + EvictScoreBoard + LoaderSolver
// + apply_far_evictions. capacity = stable slots per GPU (460). When oracle is
// non-null the sim's recomputed solve-time (cached0,cached1) is diffed against it
// (assertion/debug only — never drives a decision). When dlog is non-null the
// per-lookup + per-eviction decision stream is captured (drift-probe analysis).
VariantResult run_variant(const VariantSpec& spec,
                          const std::vector<LayerInstance>& layers,
                          int num_tokens,
                          const gpu_loader::LoaderConstants& k,
                          int capacity,
                          const OracleMap* oracle = nullptr,
                          DecisionLog* dlog = nullptr);

// ── EP-combine compute-owner dedup (FIDELITY to the force-ON engine dedup) ────
// The engine's dispatch_moe_all_ranks runs EVERY TP GPU over its OWN resident
// routed subset and combines the partials with a cross-GPU SUM that assumes each
// routed expert is computed on EXACTLY ONE GPU (disjoint EP). When the ACT reroute
// makes a routed expert resident on >1 GPU (a cross-GPU duplicate) the SUM double-
// counts it — the TD-MOE-EP-COMBINE-RESIDUAL bug. The daemon now UNCONDITIONALLY
// dedups via daemon::dedup_ep_residency (lowest-rank canonical owner) before the
// combine. This helper REUSES that exact pure header to give the sim's per-token
// cost model the SAME single owner per routed expert:
//
//   resident(slot, gpu) : post-fetch residency-AND-readiness of routed slot on gpu
//   target              : the naive home/reroute target per slot (fallback when a
//                         slot is resident on NO gpu, e.g. a reserve failure)
//   *dup_count          : #routed slots that were resident on >1 GPU (deduped)
//
// Returns the canonical compute owner (lowest-rank holder) per routed slot, so the
// LoaderSolver::evaluate cost model single-counts each routed expert exactly as the
// engine's deduped EP combine does. Pure accounting — the cache is NOT mutated
// (dedup is a compute/combine concern, not a slot concern; both copies stay
// cached). On DISJOINT placement (baseline e%tp) it is a byte-identical no-op
// (owner == target, dup_count == 0).
std::vector<int> ep_combine_owners(
    int num_slots, int num_gpus,
    const std::function<bool(int /*slot*/, int /*gpu*/)>& resident,
    const std::vector<int>& target, int* dup_count = nullptr);

// Clean-LRU CEILING (local CacheSim, no reroute): the diagnostic upper bound on
// hit-rate for a workload. Reported, not the primary metric.
double clean_lru_ceiling(const std::vector<LayerInstance>& layers, int M,
                         int capacity);

// Helpers.
int    bank_slot_of_node(const gpu_loader::LoaderConstants& k, int node);
double mean_bank_egress_us(const gpu_loader::LoaderConstants& k);

inline constexpr double kEvictWeight = 0.1;   // γ horizon discount (LS_LOADER_EVICT_WEIGHT default)
inline constexpr double kDecay       = 0.98;  // per-token recency decay (LS_EVICT_DECAY)

}  // namespace layerstorm::offline_sim
