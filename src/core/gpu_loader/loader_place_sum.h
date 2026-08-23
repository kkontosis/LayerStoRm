#pragma once
// Weighted-sum place_cons cost model (TASK short-horizon residency reframe).
//
// The CORRECT cost-model architecture for the never-lose CPU-offload decision:
// the consequence of placing a routed expert on the host CPU device is a WEIGHTED
// SUM of independent, individually-tunable factors, each in the solver's µs cost
// space so it composes directly with the fetch/compute wall. Higher cpu_place_cost
// ⇒ MORE EXPENSIVE to offload this expert to CPU (0 ⇒ free to offload).
//
//   cpu_place_cost[i] = resid_tax                          (constant churn HEURISTIC)
//                     + w_resid   · resid_baseline[i]      (recur=1 protect HEURISTIC)
//                     + w_numa    · numa_penalty[i]        (home-node fact — TRAINED)
//                     + w_hotness · hotness[i]             (freq/EMA HEURISTIC)
//                     + hbm_offload_base · home_hbm[i]     (node-based bias, STATIC)
//                     + ddr_offload_penalty · (1-home_hbm) (node-based bias, STATIC)
//                     + offnode_protect · (1-target_rare[i])        (e: complement PROTECT)
//                     − target_rare_encourage · target_rare[i]      (e: target-rare ENCOURAGE)
//
// (d) NODE-BASED offload-DECISION bias (predictor-FREE; the arena HOME NODE of an
//     expert is a static fact, not a prediction). The M3 arena placement homes the
//     globally-HOTTEST experts on the HBM banks (nodes 4-7) and the cold tail on the
//     DDR banks (nodes 0-3), so the home node is a stable global-frequency PROXY. The
//     VRAM-reallocation hypothesis (2026-08-10): OFFLOADING the HBM-home (M3-hot)
//     experts is CHEAP — their weights read from ~1 TB/s HBM and their host FFN hides
//     under the η=1.0 early-kick — and it FREES their scarce pinned GPU-VRAM so the
//     cold tail can finally cache (fewer cold-tail fetches). So HBM-home gets a LOW
//     offload cost (hbm_offload_base, ENCOURAGE) and DDR-home a HIGH one
//     (ddr_offload_penalty, PROTECT the tail that is already fetch-bound). This is
//     DISTINCT from (b) w_numa routing-locality: (b) prices the cross-node HOST read
//     of an already-decided offload; (d) is the offload DECISION bias by home-node
//     hotness. Env LS_LOADER_PLACE_HBM_OFFLOAD_BASE / _DDR_OFFLOAD_PENALTY; both
//     default 0 ⇒ inert. General no-HBM-box principle: bias offload toward one/some
//     "cold" banks and protect the rest; here the split is HBM-vs-DDR home.
//
// (e) TARGET-NODE RARE offload steering (predictor-FREE; two STATIC facts ANDed —
//     the arena HOLDING node and the M3 rarity rank, neither a prediction). Targets
//     the residency-SAFE + host-LIGHT + high-per-expert-saving regime the HBM-hot
//     (d) arm missed. An expert is target_rare = 1 iff (i) its REAL arena holding
//     node (location_node) == a chosen GPU-FREE DDR node (LS_LOADER_PLACE_OFFLOAD_
//     NODE — its experts are the MOST EXPENSIVE to GPU-fetch: cross-node to every
//     GPU) AND (ii) it is globally RARE (normalized M3 hotness < rarity_thresh —
//     INVERTED hotness: rarer ⇒ few-per-token ⇒ small host FFN that HIDES, and a
//     miss anyway ⇒ offloading it evicts NO would-be hit ⇒ residency-safe). The
//     never-lose MISS-only greedy then steers offloads to exactly this set: the
//     complement pays offnode_protect (HIGH ⇒ don't offload the hot / GPU-adjacent
//     experts — those were the (d)/Milestone-C loss), and the target-rare set is
//     optionally driven toward the 0 floor by target_rare_encourage (removes the
//     resid_tax barrier for a set KNOWN residency-safe). Paired with a CPU device
//     ON the target node, its weights read LOCALLY (fast). Env LS_LOADER_PLACE_
//     OFFNODE_PROTECT / _TARGET_RARE_ENCOURAGE; both default 0 ⇒ inert. The node id
//     + rarity threshold live dispatch-side (they resolve the {0,1} indicator).
//
// TRAINABLE vs HEURISTIC (design decision 2026-08-10). A weight on a PREDICTION
// cannot be fixed by training when the prediction is wrong: w_resid·recur zeroes on
// the recur=0 churners (b0_prev mis-marks them safe ⇒ resid_baseline=0 ⇒ no w_resid
// value prices them — measured −13.7% with any trained w_resid), and freq/EMA is
// likewise a prediction. So the two PREDICTIVE terms are HEURISTICS (env-tunable,
// NOT trained), and only w_numa — a STATIC home-node FACT, not a prediction — is a
// trainable I8 calibration param (LoaderConstants::place_sum_weights).
//
// (a) RESIDENCY — a CONSTANT expected-churn OFFLOAD-TAX (resid_tax, µs) applied to
//     EVERY offloaded candidate. This is the OPERATIVE residency heuristic: it does
//     NOT zero for recur=0, so the never-lose greedy declines the mis-priced recur=0
//     churners (the honest floor — decline-unless-clearly-beneficial: an offload must
//     save MORE than the expected cost of the future cold fetch it causes). Env
//     LS_LOADER_PLACE_RESID_TAX; default = one cold expert fetch (~960 µs). An
//     OPTIONAL w_resid·resid_baseline protect-term additionally prices the recur=1
//     set (a would-be-HIT the FREE b0_prev prev-round-union predicts) — heuristic,
//     reinforcing; the constant tax is what makes never-lose correct on the churners.
//
// (b) NUMA home-locality — per-node CPU columns. Cheap for the expert's arena home
//     node, expensive cross-node (the host FFN reads the expert weights from the
//     home DDR bank; a cross-node read is a slower draw). A STATIC topology fact ⇒
//     the ONLY trainable place_sum weight (fit black-box against the honest sim).
//
// (c) freq/EMA hotness — a high-frequency (hot) expert is expensive to offload
//     (it recurs and its GPU residency is load-bearing). HEURISTIC weight; may be 0.
//
// GPU-device columns are the neutral Phase-2 hotness/reuse-protection term
// (loader_place_hotness.h, add_hotness_place) — keeping an expert ON its GPU is
// the protection; this module prices the CPU-offload alternative against it.
//
// Pure, header-inline for the per-candidate primitives + one compiled batch
// helper (loader_place_sum.cpp). CUDA-free (layerstorm_core / INV-GPU-1). All
// live-state reads (residency, prev-round union, home node, freq) happen in the
// caller (dispatch_loader.cpp); this unit is unit-testable with no daemon state.
#include <span>
#include <vector>

namespace layerstorm::gpu_loader {

// CPU-column place_cons sum weights. resid_tax + w_resid + w_hotness are HEURISTICS
// (env LS_LOADER_PLACE_RESID_TAX / _W_RESID / _W_HOTNESS); only w_numa is a trainable
// I8 calibration param (env LS_LOADER_PLACE_W_NUMA overrides the trained value).
// Defaults: resid_tax = one cold fetch (~960 µs, the operative churn floor), w_resid
// = 1 (recur=1 protect-term), numa/hotness 0, node-bias (hbm_offload_base /
// ddr_offload_penalty) 0. All-zero ⇒ cpu_place_cost == 0 ⇒ the raw wall model
// (byte-identical to the pre-residency never-lose greedy).
struct PlaceSumWeights {
  double resid_tax = 960.0;   // HEURISTIC: constant per-offload expected-churn tax (µs)
  double w_resid   = 1.0;     // HEURISTIC: weight on the recur=1 protect-term
  double w_numa    = 0.0;     // TRAINED (I8 calib): static home-node cross-read penalty
  double w_hotness = 0.0;     // HEURISTIC: weight on the freq/EMA hotness prediction
  // (d) NODE-BASED offload-decision bias (STATIC home-node fact, env LS_LOADER_PLACE_
  // HBM_OFFLOAD_BASE / _DDR_OFFLOAD_PENALTY). Both default 0 ⇒ node term inert.
  double hbm_offload_base    = 0.0;  // µs added to HBM-home (M3-hot) — LOW = encourage
  double ddr_offload_penalty = 0.0;  // µs added to DDR-home (cold tail) — HIGH = protect
  // (e) TARGET-NODE RARE offload steering (env LS_LOADER_PLACE_OFFNODE_PROTECT /
  // _TARGET_RARE_ENCOURAGE). Both default 0 ⇒ (e) inert.
  double offnode_protect       = 0.0;  // µs added to the COMPLEMENT (protect non-target/hot)
  double target_rare_encourage = 0.0;  // µs subtracted from target-rare (encourage → 0 floor)

  bool all_zero() const {
    return resid_tax == 0.0 && w_resid == 0.0 && w_numa == 0.0 && w_hotness == 0.0 &&
           hbm_offload_base == 0.0 && ddr_offload_penalty == 0.0 &&
           offnode_protect == 0.0 && target_rare_encourage == 0.0;
  }
};

// Per-candidate factor inputs, all in the solver's µs cost space.
struct PlaceSumFactors {
  double resid_baseline = 0.0;  // recur_prob · fetch_us (would-be-HIT next-round fetch)
  double numa_penalty   = 0.0;  // cross-node host-read penalty scale (0 if home == cpu node)
  double hotness        = 0.0;  // freq-prior protection scale (hot expert = expensive)
  double home_hbm       = 0.0;  // 1 if arena home node is HBM (M3-hot), 0 if DDR (cold tail)
  double target_rare    = 0.0;  // (e) 1 iff location_node==target GPU-free node AND rare
};

// Residency / future-miss baseline (factor a). An expert is CHEAP to offload
// (0) if it would-miss-anyway next round (recur_prob 0), EXPENSIVE (recur_prob ·
// fetch_us) if it would be a cache HIT (recurs within the retention window). The
// price is the fetch it would cost cold next round if we exclude it from the GPU
// cache now. recur_prob is clamped to [0,1]; fetch_us to >= 0.
inline double residency_baseline(double recur_prob, double fetch_us) {
  if (recur_prob < 0.0) recur_prob = 0.0;
  if (recur_prob > 1.0) recur_prob = 1.0;
  if (fetch_us < 0.0) fetch_us = 0.0;
  return recur_prob * fetch_us;
}

// The CPU-device column place cost = constant churn tax + the weighted factor sum,
// clamped >= 0 (a place cost never REWARDS an offload — the wall model already prices
// the fetch/compute the offload relieves; this term can only make an offload less
// attractive). resid_tax is added to EVERY candidate (it does not depend on the
// recur prediction), so the never-lose greedy declines the recur=0 churners a
// prediction-weighted term would zero on.
inline double cpu_place_cost(const PlaceSumWeights& w, const PlaceSumFactors& f) {
  // (d) node-based bias: HBM-home pays hbm_offload_base (LOW ⇒ encourage), DDR-home
  // pays ddr_offload_penalty (HIGH ⇒ protect). home_hbm ∈ {0,1} selects the column.
  const double node_bias = w.hbm_offload_base * f.home_hbm +
                           w.ddr_offload_penalty * (1.0 - f.home_hbm);
  // (e) target-node rare steering: PROTECT the complement (+offnode_protect on
  // everything NOT target-rare), ENCOURAGE the target-rare set (−target_rare_
  // encourage, clamped at the 0 floor below ⇒ never rewards, only removes barriers).
  const double target_bias = w.offnode_protect * (1.0 - f.target_rare) -
                             w.target_rare_encourage * f.target_rare;
  const double c = w.resid_tax + w.w_resid * f.resid_baseline +
                   w.w_numa * f.numa_penalty + w.w_hotness * f.hotness + node_bias +
                   target_bias;
  return c > 0.0 ? c : 0.0;
}

// HOT PATH batch helper: out[i] = cpu_place_cost(w, factors[i]) for every
// candidate. out is resized to factors.size(). Used by the never-lose post-pass
// to price each routed miss's CPU offload before the greedy.
void compute_cpu_place_costs(const PlaceSumWeights& w,
                             std::span<const PlaceSumFactors> factors,
                             std::vector<double>& out);

}  // namespace layerstorm::gpu_loader
