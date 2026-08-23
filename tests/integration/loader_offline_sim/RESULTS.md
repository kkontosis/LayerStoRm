# Loader offline-sim — results

CPU-only, GPU-free replay that drives the **REAL engine eviction**:

- a CPU-constructible `memory::ExpertCache` (460 stable slots/GPU, `NullBackends`),
- the `gpu_loader::EvictScoreBoard` registered as the cache's `ResidencyListener`
  (membership + always-on per-layer recency + optional `LS_EVICT_DECAY`),
- the **SHARED** `gpu_loader::apply_far_evictions` victim selector — the SAME
  function the daemon's `handle_fetch_and_run_moe` calls,
- the real `gpu_loader::LoaderSolver` for ACT placement/reroute,
- the real T(j) cost model (`LoaderSolver::evaluate`) for the per-token time.

Reproduce: `cmake --build build --target loader_offline_sim_test -j` then
`ctest --test-dir build -R loader_offline_sim --output-on-failure`.

## CACHE-AFFINITY ROUTING BAKE-OFF (fab2 `285fe840`, 2026-07-12) — keeper52 affinity ported to the NVFP4 EP=2 keeper

keeper52's `KEEPER52_AFFINITY` cache-affinity routing (hits run where resident;
misses balance the per-layer fetch load, tie-broken by the cross-GPU
globally-oldest evictable victim so the per-GPU LRUs approximate ONE pooled
global LRU) was ported to the NVFP4 DeepSeek keeper
(`RoutedExpertLruFullFitDirectTest.HundredTokenDecodeFetchAndRun_FullFit_EP2`,
2×5090 TP=2/EP=2, full-pool) behind a new env gate **`KEEPER_AFFINITY=1`**
(default OFF = e%tp byte-unchanged). 4-mode side-by-side, ONE build
(`285fe840`), idle box, 2 runs/mode — per-mode misses **bit-identical** across
runs (det-reduce still holds):

| mode | env | hit | misses | tok/s (2-run) |
|---|---|---|---|---|
| baseline (e%tp) | (none) | 0.3640 | 29512 | 5.712 / 5.758 |
| plain-ACT NEW | `LS_LOADER_SHADOW=1 LS_LOADER_ACT=1 LS_EVICT_LRU_FALLBACK=1` | 0.3013 | 32418 | 6.065 / 6.004 |
| ACT+decay NEW | same + `LS_EVICT_DECAY=0.98` | 0.3014 | 32413 | 6.020 / 6.030 |
| **affinity (new)** | `KEEPER_AFFINITY=1` | **0.3643** | **29497** | **6.321 / 6.318** |

(Legacy hash-order variants excluded by design. Reminder: `LS_LOADER_ACT=1`
alone is inert in this harness — `LS_LOADER_SHADOW=1` gates
`gpu_loader.enabled` (DEBUG.md pitfall); legacy variants would be the same env
sets with `LS_EVICT_LRU_FALLBACK=0`.)

**Verdicts:**
1. **Affinity WINS tok/s on 2 GPUs: +10.5% over e%tp baseline (6.32 vs 5.73
   mean) and +4.7% over plain-ACT NEW (6.32 vs 6.03)** — the fastest routing
   mode measured on this keeper.
2. **The pooling win is NOT hit-rate on EP=2**: affinity's hit 0.3643 ≈
   baseline 0.3640 (Δ−15 misses of 46400 lookups). Two pooled ~460-slot LRUs
   barely beat two partitioned ones on this workload; the tok/s win comes from
   PLACEMENT — running hits where resident (zero same-layer relocation) and
   argmin-balancing miss fetches across the two PCIe links, where e%tp's static
   home split leaves per-layer fetch skew. ACT trades the other way: better
   balance via solver reroute but −0.06 hit (more misses), netting below
   affinity.
3. **Evict-map fidelity under affinity is exact**: TD-FAR-EVICT
   provided-victim honored=28589 rejected=0 local-fallback=0 (the test-side
   pooled-LRU model and the daemon cache stayed in lockstep).
4. **Epoch shift (context, not a regression of this A/B):** absolute numbers
   sit below the eb3fc650 anchors (baseline then 6.945/0.6663, now
   5.71–5.76/0.3640) — the fab2 code state + re-stamped 9.69.0 prepacked set
   produce a different (less cacheable) routing trajectory. All four modes
   here share ONE build/box/day, so the ranking is internally valid; the
   historical anchors are cross-epoch and only indicative.

## CANONICAL-TRAJECTORY REFRESH (fab1-sim-fixture-refresh, 2026-06-30, eb3fc650) — AUTHORITATIVE

The fixture was re-captured from the engine with **ALL THREE EP fixes landed**
(placement-invariant canonical combine + force-ON dedup + zero-resident per-slot
zero-fix) plus `compute.deterministic_reduce` default-ON. This **retires the 0.40
EP-combine-OFF epoch** below: that trace was captured BEFORE the canonical combine,
when expert placement (reroute/decay) perturbed the bf16 EP-combine `bf16(bf16(pA)+
bf16(pB))` → a routing butterfly the open-loop sim could not predict (ACT+decay
0.6291, ~0.24 short). **With the canonical combine, placement is routing-INERT: all
five scenarios converge to the ONE trajectory** (sha `34b429e3` = the on-disk
`routed_trace.csv`). The variants now differ ONLY on the EVICTION axis (recency-LRU
vs hash) — the honest cache-quality A/B.

**Trajectory identity (verified, not assumed):**
- New `routed_trace.csv` routing == canonical `base.drift` (sha `34b429e3`):
  **0 / 5800 layer-visits mismatch** (ordered, all 8 experts/visit).
- baseline == plain-ACT == ACT+decay routing: identical md5 (`607fd611…`).
- All 5 scenarios produce the identical generated token-ID stream (loader-OFF baseline,
  shadow baseline, plain-ACT, ACT+decay, both legacies).

**Engine keeper (eb3fc650, 2 idle-checked full-pool runs each, misses bit-identical):**

| variant            | hit     | misses | tok/s (2-run)   |
|--------------------|---------|--------|-----------------|
| baseline (e%tp)    | 0.6663  | 15485  | 6.945 (loader-OFF); 6.858/6.813 shadow |
| plain-ACT NEW      | 0.6548  | 16017  | 7.021 / 7.036   |
| plain-ACT legacy   | 0.2573  | 34461  | 5.154 / 5.124   |
| ACT+decay NEW      | 0.6550  | 16010  | 7.028 / 7.017   |
| ACT+decay legacy   | 0.2578  | 34438  | 5.088 / 5.148   |

**Sim vs engine on the ONE trace (open-loop, `C = 85.948 ms/token`, baseline pinned 6.945):**

| variant            | sim hit | engine hit | hit err  | sim tok/s | engine tok/s | tok/s err |
|--------------------|---------|------------|----------|-----------|--------------|-----------|
| baseline           | 0.6663  | 0.6663     | **0.0000** (cached_div=0) | 6.945 *(anchor)* | 6.945 | 0 |
| plain-ACT NEW      | 0.6548  | 0.6548     | +0.0000  | 7.167     | 7.029        | +2.0%     |
| plain-ACT legacy   | 0.2574  | 0.2573     | +0.0001  | 5.557     | 5.139        | +8.1%     |
| ACT+decay NEW      | 0.6550  | 0.6550     | −0.0000  | 7.132     | 7.023        | +1.6%     |
| ACT+decay legacy   | 0.2583  | 0.2578     | +0.0005  | 5.577     | 5.118        | +9.0%     |

**What changed / verdicts:**
1. **Baseline EXACT, cached_div=0** — the foundational gate (sim 0.6663 == engine 0.6663).
2. **The butterfly is GONE.** ACT+decay sim 0.6550 == engine 0.6550 (was 0.39 vs 0.6291,
   0.24 short). The open-loop sim now reproduces ALL FIVE variants to **<0.001** — because
   the canonical combine removed the placement→trajectory fork (TD-MOE-EP-COMBINE-FPDRIFT
   RESOLVED). The old "0.629 decay win" was the butterfly artifact; with it fixed, decay is
   genuinely a hit-rate no-op in BOTH engine and sim (ACT+decay ≈ plain-ACT, ±0.0002).
3. **The clean same-trajectory eviction A/B** (the deliverable): recency-LRU 0.6548 ≫
   hash-order 0.2573 on the IDENTICAL trajectory — a ~0.40 gap, reproduced by the sim
   (0.6548 vs 0.2574). The board's recency ranking is the dominant cache-quality lever;
   placement is now routing-inert. Legacy faithfulness comes from the SHARED
   `memory::ExpertCache` hash iteration + `apply_far_evictions` (same class, both sides).
4. **tok/s ranking now CORRECT** — sim ACT (7.13–7.17) > baseline (6.945) > legacy (5.56–5.58),
   matching the engine; the retired epoch mis-ordered the top (butterfly). Magnitude within
   ≤9% (legacy a touch high; the cost model is still myopic, INV-LOADER-OBJECTIVE-MYOPIC).
5. **EP-combine dedup** — disjoint baseline a byte-identical no-op (`ep_combine_dups=0`).
   On the more-cacheable trace ACT now co-resides a FEW routed duplicates at compute time
   (plain-ACT 129, decay 144, legacies 4/21 of 46400) — still ≪ lookups, so the deduped
   cost model is numerically indistinguishable from pre-dedup HERE (the dedup synthetic
   regime is covered by the `EpCombineOwners` cases). dupRes% ~0.2% (< 1%).

Build note: this campaign also flipped `LAYERSTORM_USE_NVME` default ON in CMakeLists
(the io_uring ArenaLoader) — fresh worktree builds were silently falling back to the
worker-pool preload (0.94 vs 3.37 GB/s, ~3.6× slower init); graceful auto-disable if
liburing is absent. No effect on decode tok/s / hit-rates (preload is one-time init I/O).

## DET-REDUCE RE-ASSESS (fab1-sim-reassess, 2026-06-28) — predictor vs a STABLE GPU [SUPERSEDED by CANONICAL-TRAJECTORY REFRESH — retired 0.40 EP-combine-OFF epoch; the "decay butterfly" it documents was the placement→bf16-EP-combine fork, now FIXED by the canonical combine]

`compute.deterministic_reduce` is now **DEFAULT-ON** (fab1 `f5d2c1ff`), so the GPU
keeper is **bit-reproducible run-to-run** — a stable target for the predictor
(previously drift-confounded). Re-measured the keeper
`RoutedExpertLruFullFitDirectTest.HundredTokenDecodeFetchAndRun_FullFit_EP2`
(2×5090 EP=2, **full pool**, NO `LS_DRIFT_SMALL_POOL`, det-reduce default), **3
idle-checked runs per variant**.

**Determinism CONFIRMED: zero hit-rate spread.** Every variant produced the
IDENTICAL miss count across all 3 runs (baseline 27832 ×3, plain-ACT 28185 ×3,
ACT+decay 17212 ×3, plain-ACT-legacy 33247 ×3, ACT+decay-legacy 33371 ×3). tok/s
spread is wall-clock jitter only (~0.5–1.3%). This is the predicted effect of
DET-REDUCE: the atomic-epoch run-to-run hit-rate noise (~0.002 plain-ACT) is gone.

**Baseline trace RE-CAPTURED.** The on-disk `routed_trace.csv` was captured under
the OLD atomic baseline; under det-reduce the baseline trajectory **diverges at
token 72 / layer 53** (a near-tie bf16 flip; identical for tokens 0–71, 17% of
rows differ overall, 1269/5800 (token,layer) cells differ). So the stale trace did
NOT reproduce the current baseline. Re-captured the ONE baseline trace from the
deterministic engine via shadow dump (`LS_LOADER_SHADOW` is behavior-neutral —
the capture run reproduced the pure baseline 0.4002 / 27832 misses exactly) and
refreshed `routed_trace.csv` + `oracle_baseline.csv`. **On the refreshed trace the
sim baseline reproduces the GPU baseline EXACTLY: 0.4002 vs 0.4002, cached_div=0**
(the foundational gate — tighter than the old 0.4006-vs-0.4015). The per-policy
`oracle_act/decay.csv` are NOT re-captured (reference-only, stale hash-order epoch;
algo-not-data forbids per-variant trace fabrication).

**Affine tok/s calibration (re-derived):** `C = 96.217 ms/token`, pinning the
baseline `T_pred=93.4 ms` to the new deterministic baseline `5.275 tok/s`
(3-run mean 5.273/5.289/5.262). `tok_s_sim(V) = 1000/(T_pred(V)+C)`.

### The deliverable — per-scenario GPU (stable) vs open-loop sim

| variant            | GPU hit (±spread) | sim hit | hit err  | GPU tok/s (range)   | sim tok/s | tok/s err |
|--------------------|-------------------|---------|----------|---------------------|-----------|-----------|
| baseline           | **0.4002** (±0)   | 0.4002  | **0.0000** | 5.275 (5.262–5.289) | 5.275 *(anchor)* | 0      |
| plain-ACT NEW      | **0.3926** (±0)   | 0.3900  | −0.0026  | 5.463 (5.458–5.473) | 5.637     | +3.2%     |
| ACT+decay NEW      | **0.6291** (±0)   | 0.3900  | **−0.2391** | 7.078 (7.048–7.127) | 5.598  | **−20.9%** |
| plain-ACT legacy   | **0.2835** (±0)   | 0.2341  | −0.0494  | 5.260 (5.241–5.271) | 5.211     | −0.9%     |
| ACT+decay legacy   | **0.2808** (±0)   | 0.2342  | −0.0466  | 5.230 (5.190–5.259) | 5.215     | −0.3%     |

(±spread on hit = literally zero: bit-identical misses across the 3 runs.)

### Hit/miss verdict — predicts baseline (exact) + plain-ACT (close); MISSES the decay butterfly
- **baseline: EXACT** (0.4002 = 0.4002, cached_div=0). The refreshed fixed-trace
  clean-LRU replay reproduces the deterministic engine bit-for-bit.
- **plain-ACT NEW: CLOSE** (−0.0026). The reroute machinery's hit-rate effect IS
  visible from the baseline trace — plain-ACT is a small fixed-trace perturbation.
- **legacy (hash-order) variants: under-predicts by ~0.047–0.049.** The legacy
  `LS_EVICT_LRU_FALLBACK=0` hash-order eviction is trace-sensitive (its victim
  picks depend on hash iteration order that the open-loop replay only approximates);
  still the correct ORDER (well below baseline).
- **ACT+decay NEW: DIVERGES by −0.2391** — the headline. On the fixed baseline trace
  decay is a hit-rate WASH (sim 0.3900 ≈ plain-ACT 0.3900), so the sim predicts
  ~0.39. The keeper gets 0.6291. With the GPU now deterministic, that 0.629 is
  **NOT a sample of a noisy distribution** (0.6291 bit-stable ×3) — it is a
  DETERMINISTIC closed-loop placement-butterfly: decay's placement change perturbs
  the bf16 cross-GPU reduction grouping → a different, more-cacheable routing
  trajectory. A fixed-baseline-trace replay cannot see that BY CONSTRUCTION. The
  0.24 gap is a real, stable prediction error, not drift noise.

### tok/s verdict — RANKING fails at the top; MAGNITUDE good except the butterfly
- The affine cost model is **anchored exact at baseline** and **within ≤3.2%** for
  plain-ACT and both legacies — so for non-decay variants the baseline-anchored
  cost model is a usable MAGNITUDE predictor of the now-stable GPU tok/s.
- It **mis-orders the top**: sim ranks plain-ACT (5.637) ≳ ACT+decay (5.598), but
  the GPU has ACT+decay far ahead (7.078). The cost model is fed the baseline-trace
  miss pattern (T_pred 82.4 ≈ plain-ACT 81.2), so it cannot know ACT+decay actually
  runs 17212 misses (vs 28185) on the GPU. Result: ACT+decay tok/s is **−20.9%
  under-predicted** — the SAME butterfly that breaks the hit-rate prediction (fewer
  real misses ⇒ shorter real makespan the open-loop model never sees). Note this is
  a DIFFERENT residual from the documented cost-model makespan bias (~1.48× under-
  predict + omitted host-launch/attention/handoff): here the model's INPUT (miss
  pattern) is wrong, not just its scale.

### Overall verdict — trustworthy open-loop for baseline + plain-ACT; GPU keeper for decay
- **Hit-rate predictor:** YES for baseline (exact) and plain-ACT (±0.003); NO for
  ACT+decay (the closed-loop butterfly). Legacy variants: ranking-correct, ±0.05.
- **tok/s predictor:** YES (magnitude, ≤3%) for baseline/plain-ACT/legacies; NO for
  ACT+decay (−21%, wrong ranking at the top).
- **Scope where it holds:** open-loop, fixed-trace, for policies whose effect is a
  small perturbation of the captured trajectory (baseline, plain-ACT). Any policy
  that seeds an autoregressive trajectory fork (decay) must fall back to the GPU
  keeper. The CPU sim is the deterministic FIRST gate; the keeper is the closed-loop
  SECOND gate — both now stable, so a single keeper run per variant suffices (no
  averaging over repeats needed; PROMPT-averaging is the only remaining caveat).

## FIDELITY FIX (fab1-offline-sim-fidelity): one trace, vary only the algorithm

**Verdict: it was a TEST-DATA bug; the engine mechanism is FAITHFUL.** All 5
variants now replay the SINGLE baseline trace `routed_trace.csv`, differing ONLY
by the algorithm. The prior epoch drove each variant off its OWN policy-drifted
oracle trace, which manufactured a spurious decay inversion (sim 0.754→0.384,
keeper 0.392→0.629). The oracle traces are NOT a small perturbation of the
baseline — their routed expert SETS differ by ~98% per (token,layer) — and they
have wildly different intrinsic cacheability (clean-LRU ceiling `oracle_act`
0.7629 vs `oracle_decay` 0.3931). The table compared apples to oranges.

## 5-variant table (single workload = routed_trace.csv)

| variant            | sim hit | keeper hit | dupRes% | reroute | rr→cached | recency | sim tok/s* |
|--------------------|---------|------------|---------|---------|-----------|---------|------------|
| baseline           | **0.4006** | 0.4015 | 0.000   | 0       | 0         | 0       | **5.249**  |
| plain-ACT NEW      | **0.3906** | **0.392** | 0.004 | 5805    | 1170      | 5800    | 5.607      |
| plain-ACT legacy   | 0.2358  | 0.282      | 0.000   | 8603    | 3903      | 5800    | 5.188      |
| ACT+decay NEW      | 0.3905  | 0.629      | 0.004   | 5967    | 1203      | 5800    | 5.566      |
| ACT+decay legacy   | 0.2343  | 0.278      | 0.008   | 8357    | 3892      | 5800    | 5.188      |

\* `tok_s_sim(V) = 1000/(T_pred(V) + C)`, one affine `C = 97.21 ms/token` pinning
baseline to the keeper 5.249 tok/s. Hit-rates are REAL (engine-reproduced);
tok/s are a baseline-anchored COST-MODEL estimate (INV-LOADER-OBJECTIVE-MYOPIC).

## Findings (evidence-first)

1. **plain-ACT NEW now MATCHES the keeper** — sim 0.3906 vs keeper 0.392, on the
   single trace. The previous 0.754 was the cacheability of `oracle_act.csv`
   (ceiling 0.76), not the algorithm. This is the headline confirmation that the
   per-policy traces, not the code, drove the divergence.

2. **LS_EVICT_DECAY is NOT a no-op — it perturbs PLACEMENT (refined by the
   drift-probe, §DRIFT-PROBE).** The earlier "no-op" wording was imprecise.
   `decay_all` is order-preserving on the per-GPU victim HEAP for non-duplicates
   (rank 0 ⇒ eff == raw, uniform scale) so it never DIRECTLY flips a non-dup
   `cheapest_keys` victim — BUT it SCALES the raw recency-clock magnitudes that
   feed `cheapest_scores_sorted` → `evict_cum` → the placement solver, flipping
   **1454 / 46 400 (3.1%) reroute targets**, cascading into **172 different
   hit/miss outcomes** and **18 378 different eviction victims** on the SAME fixed
   trace. The HIT-RATE is nonetheless a near-wash (ACT+decay NEW 0.3905 vs
   plain-ACT NEW 0.3906; legacy 0.2343 vs 0.2358): decay reshuffles placement
   without improving cacheability. Asserted as a hit-rate near-wash in
   `loader_offline_sim_test` (`r[3]≈r[1]`, `r[4]≈r[2]` within 0.005) and pinned
   exactly by `decay_drift_probe_test`.

3. **The placement-duplicate hypothesis is DISPROVEN.** Cross-GPU duplicate
   residency is ≈ 0.004% and ≤ 2 duplicate inserts over 46400 lookups for every
   ACT variant. Reason: the `LoaderSolver` sees the cached mask and routes a miss
   to a GPU where the expert is already resident (free transfer), so reroutes
   either land on a cached GPU (a free hit — `rr→cached`) or fetch a single fresh
   copy after the home copy was already evicted; they do NOT co-reside. So ACT
   placement does NOT mint duplicates.

4. **The keeper's decay gain (0.392→0.629) is ROUTING-DRIFT-generated** (honest
   dual outcome B). It was measured WITH the always-on recency clock already
   present (commit 55a5c85f, 2026-06-26 — same commit that introduced the clock
   AND recorded the keeper A/B). Since decay is a fixed-workload no-op (finding 2)
   and placement mints no duplicates (finding 3), the only remaining source of the
   0.24 gap is the closed-loop autoregressive workload divergence. A fixed-trace
   CPU replay cannot reproduce that BY CONSTRUCTION. **For the decay metric, FP
   routing-drift is NOT negligible.**
   **CORRECTED MECHANISM (`fab1-decay-rootcause`, 2026-06-28, `spec/I8_INVESTIGATION_REPORT.md` §9):**
   it is NOT "a different eviction SKIPS different experts → different MoE numerics" — at FullFit
   nothing is skipped (FETCH_AND_RUN fetches every routed expert with a 5 s timeout before finalize;
   all compute). The actual seed: decay scales `evict_cum` → the `LoaderSolver` picks a different
   device for a MISS expert → ACT REROUTES it off `e%tp` (`dispatch_loader.cpp:271`) → the EP
   partition changes → each GPU's `finalize_moe_routing_bf16` rounds its per-token PARTIAL sum (over a
   placement-determined subset) to bf16 → the combined `bf16(bf16(pA)+bf16(pB))` is bf16-ULP
   placement-dependent → near-tie gating flip → trajectory fork. DECISIVE: forcing a canonical `e%tp`
   partition (`LS_LOADER_FORCE_IDENTITY`) collapses the keeper ACT+decay 0.6291→0.4002 BYTE-IDENTICAL
   to baseline (27832 misses). So the 0.24 is a placement→bf16-EP-combine butterfly, fixable by a
   Phase-1b fp32/canonical EP combine (TD-MOE-EP-COMBINE-FPDRIFT), not an eviction-quality gain.

5. **Recency-magnitude trap (TD-EVICT-RECENCY-MAGNITUDE) confirmed and refined.**
   The recency clock reaches ~5800, so the additive duplicate penalty
   `max(0, raw − rank·0.9)` is noise: even where a duplicate exists it is not
   evicted first. The DEBUG.md belief that "decay keeps raws O(base)" is FALSE —
   touches re-stamp raw to the growing clock, so 0.98×/token cannot bring raws
   near 0.9 in a 100-token run (would need ~435 untouched tokens). Decay can bite
   neither via ordering (no-op) nor via the duplicate penalty (swamped).

## DRIFT-PROBE (fab1-evict-drift-probe): resolving the decay "no-op" contradiction

`decay_drift_probe_test` (CPU-only, REAL board+cache) resolves the contradiction
the prior agent's own numbers betrayed (a true no-op ⇒ bit-identical runs; the
keeper shows ACT+decay 0.629 vs plain-ACT 0.392, and even the fixed-trace 0.3905
≠ 0.3906). Four evidence parts:

**A. Decision-stream diff (decay-on vs decay-off, fixed `routed_trace.csv`):**
- lookups with a different reroute TARGET: **1454 / 46 400 (3.13%)**
- lookups with a different HIT/MISS: **172 (0.37%)**
- reroute count: 5805 → 5967 (**Δ +162**)
- (token,layer,gpu) cells with a different victim set: **7146**; total victim keys
  that differ (symmetric diff): **18 378**
- FIRST divergence: `(t=6,l=48,e=207)` — a PLACEMENT flip (plain target=0, decay
  target=1; both miss). The seed is placement, not victim ordering or a hit flip.
- => decay is decisively **NOT bit-identical** (not a no-op), yet the net
  hit-rate barely moves (0.3906 vs 0.3905).

**B. Mechanism on the REAL `EvictScoreBoard` (concrete values):**
- (1) `decay_all(0.98)` changes `cheapest_scores_sorted` VALUES 100/200/300 →
  98/196/294 — i.e. it scales the `evict_cum` MAGNITUDES fed to the placement
  solver (a real input). (2) it leaves non-dup victim ORDER invariant
  (eff==raw, uniform scale). (3a) WITH a cross-GPU DUPLICATE it CAN flip a victim
  (raw gap in (base, base/f): `eff = raw − rank·base` is non-linear in raw — pre
  victim 60, post victim 50). (3b) the `max(0, …)` clamp floors a trivialized
  duplicate's eff at 0 (ties broken by slot index).
- => decay's fixed-trace seed is the `evict_cum` MAGNITUDE channel (drives the
  1454 placement flips) plus the rare duplicate/clamp DIRECT victim flip (only ≤2
  duplicates exist here, so negligible). The mechanism is **algorithmic
  (placement-cost scaling), NOT an FP/clamp accident**.

**VERDICT 1 — decay is an algorithmic lever on PLACEMENT, but a butterfly SEED for
hit-rate.** It acts on a real signal (the eviction-cost magnitude in the placement
cost model), so it is not a numerical accident; but on a fixed trace that lever
nets to a hit-rate wash (it even adds 162 reroutes). The keeper's +0.24 is the
autoregressive amplification of these placement perturbations into a different
routing trajectory with higher intrinsic cacheability (oracle ceilings 0.39 vs
0.76) — i.e. the 0.629 is a drift-amplified butterfly artifact, even less reliable
than "drift" because its *sign* is set by chaotic amplification, not by the
eviction signal decay nominally acts on. (Corrects the prior "no-op" wording and
sharpens TD-EVICT-DECAY-KEEPER-DRIFT.)

**C. LRU-correctness probe + bounded-recency prototype:**
- plain-ACT NEW = **0.3906** vs the textbook clean-LRU (no-reroute) ceiling =
  **0.4006** → a **0.0100 gap**: the ACT reroute machinery leaves DETERMINISTIC
  headroom (it is WORSE than a plain per-GPU LRU). This is recoverable headroom,
  not workload-bound — the user's LRU-correctness thesis holds.
- bounded-recency prototype (TD-EVICT-RECENCY-MAGNITUDE, sim-local via
  `update_existing_only`, ZERO production change): replacing the unbounded clock
  (→5800) with a counter incremented by `base/cap`·{0.1,1,10} per layer-visit
  raises plain-ACT to **0.3936 / 0.3941 / 0.3942** (gap 0.0100 → 0.0064–0.0070)
  and cuts reroutes 5805 → ~4700, DETERMINISTICALLY. Bounding earns ~⅓ of the
  ceiling gap reliably (no drift needed) and confirms the over-rerouting is driven
  by the inflated recency MAGNITUDE; the residual ~0.006 is the reroute itself.

**VERDICT 2 — the NEW eviction is NOT a faithful clean-LRU; there is real
deterministic headroom (0.01).** Bounding the recency clock recovers ~⅓ of it
reliably (the constructive payoff); the rest is intrinsic to rerouting.

**D. Determinism:** the CPU sim is bit-deterministic (two plain-ACT runs →
identical 0.390647). Decay's influence is causal STEERING on a fixed seed, not
run-to-run sampling. On the GPU engine the trajectory selection rides on SM120
CUTLASS/NCCL reduction non-determinism (TD-F-2-bx ~25–30% top1_prob swings,
INV-IPC-5, TD-F-5-eq), so the keeper's 0.392/0.629 are likely TWO SAMPLES from a
nondeterministic trajectory distribution rather than a clean deterministic
decay→0.629 map. A single idle-checked GPU repeat would measure the spread; not
run here (GPU secondary; CPU analysis primary).

## What is validated (high confidence)

1. **baseline — EXACT.** Real cache + clean-LRU orchestrator map reproduces the
   recorded engine bit-for-bit: hit 0.4006 (keeper 0.4015), **0 cached
   divergences** across all 46 400 rows; calibrates to exactly 5.249 tok/s.
2. **plain-ACT NEW reproduces the keeper** (0.3906 vs 0.392) on the single trace.
3. **decay no-op + zero placement-duplicates** asserted as the mechanism truth.

## Production impact

NONE. The fix is in the TEST + sim harness only (single trace, instrumentation,
verdict assertions). The shared `far_evict` / daemon path and the
`EvictScoreBoard` / `ExpertCache` are UNCHANGED — no keeper re-validation needed.

The real production concern surfaced (not changed here) is TD-EVICT-RECENCY-
MAGNITUDE: `LS_EVICT_DECAY` does not do what it appears to; its keeper benefit is
an unstable drift artifact, not an algorithmic improvement. Bounding the recency
clock or making the duplicate penalty multiplicative is the follow-up (filed).

## Mechanism the sim drives (faithful to the engine)

Per MoE layer-visit it mirrors `handle_fetch_and_run_moe` + `route_moe_by_loader`:
residency check at the `e%tp` home → `advance_recency` + `touch_existing` (+
`decay_all` per token) → `LoaderSolver::solve` (with `evict_cum` from the board's
`cheapest_scores_sorted`) → reroute misses → lock cached → **`apply_far_evictions`
(shared helper)** for the make-room step (orchestrator clean-LRU victim map honor
+ board `cheapest_keys` / hash-order fallback) → reserve+`mark_all_ready` the
misses (the board learns residency via the listener). A miss = an H2D transfer =
not resident at the EXECUTED target gpu (the keeper's `1 − transfers/lookups`).

## EP-COMBINE DEDUP FIDELITY (fab1-sim-dedup, 2026-06-29) — modeled, mirrors the engine

**VERDICT: the sim MODELS the EP/duplicate dimension, and now mirrors the engine's
force-ON cross-GPU EP-combine dedup at its per-token compute/combine accounting.**

The sim drives a REAL per-GPU `memory::ExpertCache` (residency tracked per GPU) and
the ACT reroute CAN make a routed expert resident on >1 GPU (it already carried a
`dup_inserts` / `dup_resident_layersum` census). So the cross-GPU duplicate the
engine's dedup (TD-MOE-EP-COMBINE-RESIDUAL fix, INV-MOE-EP-DISJOINT) targets is
representable here — this is the MODELED path, not a fidelity gap.

What was missing: the cost model `LoaderSolver::evaluate` is single-count by
construction (one `exec[]` device per routed slot), so the engine's *pre-dedup*
"every holder computes → 2·c_k" double-count never existed in the sim. What
diverged from the engine was the OWNER CHOICE — the sim computed each routed
expert at its naive home/reroute `target`, whereas the deduped engine computes it
at the LOWEST-rank holder among its resident GPUs.

**Fix:** `offline_sim::ep_combine_owners()` REUSES the pure
`daemon::dedup_ep_residency()` header (the same one `dispatch_moe_all_ranks` calls)
over per-GPU routed-expert residency bitsets to pick each routed expert's compute
owner = lowest-rank holder; `run_variant` feeds those owners to `evaluate`
(replacing `exec=target`). Pure accounting — the cache is NOT mutated (dedup is a
compute/combine concern, not a slot concern), so the **hit-rate is unchanged**.

**What it changed (evidence):**
- **baseline still EXACT** — disjoint e%tp ⇒ zero routed cross-GPU duplicates and
  zero compute-owner shifts (`ep_combine_dups==0`, `ep_owner_shifts==0`) ⇒ a
  byte-identical no-op ⇒ baseline reproduces 0.4002 / cached_div=0 / 5.275 tok/s
  exactly (asserted).
- **ACT variants: empirically negligible HERE** — on the fixed trace the
  LoaderSolver routes misses onto already-cached GPUs (free hits), so routed
  duplicates essentially never co-reside (`ep_combine_dups ≪ lookups/1000`,
  asserted) ⇒ the deduped cost model is numerically indistinguishable from the
  pre-dedup one on this workload. The dedup matters only for placement policies
  that DO co-reside routed duplicates — which this trace does not exhibit.

**Test:** 4 new `EpCombineOwners` CPU cases mirror `tests/unit/ep_residency_dedup_test.cpp`
at the sim owner-mapping level (disjoint = no-op; a synthetic cross-GPU duplicate
collapses to the lowest-rank owner; expert on ALL GPUs → rank 0; non-resident slot
falls back to target = still counted once), so the double-count-correction regime
is exercised synthetically even though the captured trace does not produce it. The
5-variant replay gains baseline no-op + ACT near-zero-duplicate assertions. All
green; INV-GPU-1 clean. No production/engine change.
