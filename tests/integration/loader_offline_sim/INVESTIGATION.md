# ACT hit-rate investigation: tiny FP drift, or placement/eviction?

**Question.** The I8 GPU-loader's ACT placement has hit-rate **0.3091** vs the
e%tp baseline **0.4006**. Is that 9-point drop caused by the tiny FP routing
drift (ACT reroutes → allreduce sums in a different FP order → gating top-K
flips → ACT routes different experts), or by placement/eviction?

All numbers below are from the committed engine oracles in `assets/`
(`oracle_baseline.csv`, `oracle_act.csv`, `routed_trace.csv`). Reproduce with
`python3 divergence_analysis.py` (Q1/Q2) and `python3 clean_lru_sim.py` (Q3/Q4).
Engine `src/**` was read but not modified.

---

## BOTTOM LINE (Q6)

**The hit-rate drop is NOT caused by the FP routing drift. It is caused
ENTIRELY by the engine's eviction under ACT.** The FP drift is real and it
makes routing diverge enormously, but that divergence — fed through a clean
460-slot LRU — would *raise* hit-rate to ~0.73-0.76, not lower it. The drop to
0.31 is an eviction artifact.

### Q3 decomposition (clean per-GPU LRU, 460 slots, key=(layer,expert))

| step | configuration | hit-rate | contribution |
|------|---------------|----------|--------------|
| **A** | baseline routing + e%tp place + clean LRU | **0.4005** | — (validates LRU model vs engine 0.4006) |
| **B** | ACT routing + e%tp place + clean LRU | **0.7625** | **routing drift = B−A = +0.362** |
| **C** | ACT routing + ACT place (j) + clean LRU | **0.7306** | **placement = C−B = −0.032** |
| **D** | engine actual (mean cached@j, oracle_act) | **0.3091** | **eviction = D−C = −0.422** |

Total D−A = −0.092 = the real 0.4006→0.3091 drop. ✓

- **Routing drift: +0.36 (HELPS).** ACT's drifted routing mode-collapses onto a
  smaller, repeated expert set → *more* cache locality, not less.
- **Placement (j vs e%tp): −0.03 (negligible).**
- **Eviction: −0.42 (dominant, and the whole story).** A clean LRU on ACT's own
  trace keeps 0.73; the engine keeps 0.31. The engine throws away 42 points of
  hit-rate that a textbook LRU would have retained.

**One-liner:** tiny drift → big *routing* change, but that change would *improve*
hit-rate; the actual drop is 100% the engine's broken eviction under reroute.

---

## Q1 — routing divergence magnitude (it is NOT tiny in aggregate)

Per-(token,layer) expert-set overlap `|base ∩ act| / 8`:

- 5702 / 5800 (token,layer)s differ — but the **magnitude** is large, not 1/layer:
- overlap **mean 1.065, median 1, min 0, max 8** of 8.
- histogram (#shared of 8): `{0:2484, 1:1934, 2:820, 3:275, 4:76, 5:36, 6:29, 7:48, 8:98}`.
- For reference, two *independent* random K=8 picks of 256 share 0.25 on average.
  ACT vs baseline share ~1.07 — i.e. **routing has almost fully decorrelated**,
  only slightly above random.

**But it starts tiny and cascades.** The first divergence is a *single near-tie
flip*:

```
token 0, layer 22:  baseline {13,18,65,213,234,241,247,248}
                    act      {13,18,65,213,    241,247,248,249}   expert 234 → 249
```

Token 0 layers 3..21 are **bit-identical** (overlap 8); token-0 mean overlap is
7.79/8 — the drift is confined to a handful of late layers. Then it amplifies:

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 40 | 80 |
|-------|---|---|---|---|---|---|---|----|----|----|
| mean overlap/8 | 7.79 | 3.43 | 2.50 | 2.16 | 1.98 | 1.19 | 0.74 | 0.97 | 0.43 | 0.98 |

**Cascade mechanism.** One near-tie gating flip at L22 perturbs the residual
stream; deeper layers in the same token flip more; then token 0's diverged
hidden states are written to the **KV cache**, so token 1 attends to drifted KV
and diverges from the start (3.43/8), and by token ~6 routing is essentially
decorrelated. This is a deterministic chaotic cascade through depth + the
autoregressive feedback loop — a textbook small-perturbation amplification.

---

## Q2 — is it really FP drift (same algorithm)?

**Yes — FP non-associativity, not a logic difference.** Evidence:

1. **The expert *workload* is policy-independent.** EP only changes *which GPU
   computes* an expert, not which experts are selected for a given hidden state.
   So baseline vs ACT differ only in the *summation grouping* of the routed-expert
   allreduce across the 2 GPUs → a pure FP-reduction-order change, same math.
2. **First-divergence signature is a single near-tie flip** (234→249, 7/8
   identical), exactly what a sub-ULP hidden-state perturbation crossing one
   top-8 decision boundary looks like.
3. **Early layers are bit-identical.** Token 0 layers 3..21 match exactly; a
   *logic* difference in placement would perturb numerics from layer 3 onward.
   Instead the trace is bit-stable until a near-tie finally tips at L22.
4. **Determinism.** Baseline reproduces exactly (the offline sim reproduces
   baseline 0.4006; the shadow capture reproduced 5.331 tok/s / 0.4006). The
   kernels and reduction order are fixed per method, so ACT is self-deterministic
   — re-runs give the same reroute decisions → the same FP order → the same
   cascade. The divergence is strictly baseline-vs-ACT, not run-to-run noise.
   (No fresh GPU capture was needed; the oracle evidence is conclusive.)

---

## Q4 — reconciling the clean-LRU numbers (0.40 vs 0.73)

Two prior diagnostics disagreed: ~0.40 ("baseline trace + solver-recomputed j")
vs 0.73 ("oracle_act trace + oracle j").

**0.73 is the correct clean-LRU figure for ACT** — it is exactly experiment **C
= 0.7306**. The **0.40 figure is the artifact.** Its bug: it fed the **baseline
expert sequence** into the simulator, not ACT's actual routed experts. But ACT
routes *almost entirely different experts* (Q1: overlap ~1/8). So the 0.40 run
measured the locality of *baseline* routing under ACT placement — it answers the
wrong question and lands near baseline's 0.40 by construction.

The "oracle `j` is circular" worry is **not material here.** Experiment **B**
(ACT routing + **e%tp** placement) uses `oj = expert%2`, which has zero
residency dependence — fully non-circular — and gives an even *higher* 0.7625.
So the high clean-LRU figure for ACT is robust and not an artifact of reusing
oracle `j`. **Correct clean-LRU(C) = 0.73 (range 0.73–0.76 depending on
placement).**

---

## Q5 — why the engine (0.31) retains far less than a clean LRU (0.73)

Reading the actual eviction path:

- **Harness** (`tests/integration/first_token_test.cpp`, ~L1149-1199): builds an
  LRU eviction map but models residency under **e%tp** placement
  (`entries[i].gpu_idx = e % tp`) and names each victim on the **home** GPU `oj`.
- **Daemon ACT reroute** (`src/daemon/dispatch_loader.cpp:222-236`): reroutes
  only **miss** experts to the solver's `j`
  (`state.experts[i].target_gpu = sp` when `!was_cached && sp != target`).
- **Evict-map consumption** (`src/daemon/dispatch_moe.cpp:1616-1639`): applies
  the supplied victim on `gpu = er.target_gpu` — the **rerouted `j`**, not the
  `oj` the harness named the victim for. The victim (LRU tail of the harness's
  `oj`-model) is usually **not resident on `j`** → `request_evict` fails →
  `far_evict_rejected_`.
- **Fallback** (`dispatch_moe.cpp:1641-1667`): "Unranked resident scan (no
  recency/score)". It calls `ExpertCache::eviction_inputs`
  (`src/core/memory/expert_cache.cpp:404-418`), which iterates an
  **`unordered_map` (hash order)**. So once the supplied victim is rejected, the
  engine evicts an **arbitrary** resident.

**Net effect:** under ACT the orchestrator-guided LRU map is structurally
mismatched to where experts actually live (miss→`j` reroute vs `oj`-model map),
so it is mostly rejected, and the engine falls back to hash-order (effectively
random, sometimes anti-LRU) eviction. It cannot retain the hot working set, so it
sees **none** of the mode-collapse locality benefit.

Confirming sims (`clean_lru_sim.py` + the random-evict probe):
- clean LRU on ACT's trace climbs with mode-collapse: per-token hit 0.40 (t3) →
  0.84 (t40) → **0.98 (t80)**.
- the **engine stays flat ~0.29-0.36 across all 100 tokens** — zero locality
  capture.
- a random-eviction sim on ACT j-placement gives 0.63 (still benefits from
  collapse); the engine's 0.31 is **worse than random**, consistent with the
  rejected-map + hash-order eviction actively displacing hot experts.

**Cross-GPU duplication is NOT the cause.** Measured from the oracle: ACT
`cachedEITHER = 0.3113` ≈ `cached@j = 0.3091`, `cachedBOTH = 0.0010`,
miss@j-but-resident-elsewhere = 0.0022. Experts are not split/duplicated across
GPUs; the engine simply evicts the wrong ones.

---

## Corrections to the brief's framing

- The brief suspected the per-layer divergence "may be tiny (~1 expert/layer)".
  **It is the opposite in aggregate** (mean overlap 1.07/8 — ~7 of 8 differ).
  It *starts* as a single flip and cascades to near-complete decorrelation.
- The brief presumed 0.73 was the methodology artifact and ~0.40 was sound.
  **Reversed:** 0.73 (= experiment C, and ≈ B = 0.76) is the correct clean-LRU
  figure for ACT; 0.40 is the artifact (wrong — baseline — expert sequence).
- Therefore the routing drift does **not** lower hit-rate; it would raise it. The
  drop is purely eviction.

---

# 2026-07-17 addendum: 4-GPU trajectory sim (trajectory_sim.py) — ACT-vs-affinity root cause

New instrument: `trajectory_sim.py` replays LS_LOADER_SHADOW_DUMP trajectories
(committed: `assets/traj_4gpu_{act_noevict,affinity}.jsonl.gz`, keeper52 EP=4
GLM-5.2, caps [203,203,443,443]) through a clean per-GPU LRU under
interchangeable placement policies and predicts times via the calibration model
(`tests/assets/gpu_loader_calibration_4gpu_hbm.json`).

## Validation
- Affinity trajectory, executed policy: sim hit **0.7370** == engine 0.7370
  (exact). Affinity re-simulation: 0.7369. The old Q3 "engine eviction −0.42"
  artifact is GONE in the current engine (clean LRU == engine).
- ACT trajectory, executed policy: sim 0.7315 vs engine 0.7190 (−1.25pp
  residual engine effect; small).
- ACT hit semantics verified: engine hit == cached[oj] OR cached[j] (a miss
  rerouted onto a resident replica is a free hit); cached-at-target alone is
  only 0.3358.

## Matrix (hit rate | predicted ms/tok via the model)

| policy    | ACT trace       | affinity trace  |
|-----------|-----------------|-----------------|
| affinity  | **0.7952 | 48.2** | 0.7369 | 55.6  |
| etp       | 0.7429 | 64.2   | 0.6738 | 75.7  |
| solver    | 0.7315 | 51.4   | 0.6515 | 73.5  |
| executed  | 0.7315 | 51.4   | 0.7370 | 55.6  |

## Conclusions
1. **Hits/misses differ between arms because the TRAJECTORIES differ**, not
   because "the same decision" is scored differently: routed expert sets share
   only 97/7500 records (mean overlap 1.27/8, barely above the 0.25 random
   floor; first flip at token 0 layer 9). Placement changes the EP-combine FP
   order → near-tie gating flips → autoregressive cascade. Within an arm the
   trajectory is bit-stable (5/5 identical hit rates).
2. **Placement effect (same trace): affinity wins everywhere.** On ACT's own
   trace affinity placement scores 0.7952 vs the solver's 0.7315 (+6.4pp). The
   solver's losses are visible mechanisms: 15.8–20.3% of its fetches are
   DUPLICATES (expert re-fetched to a second GPU while resident elsewhere),
   1.7× the evictions-of-soon-reused, higher multi-home churn.
3. **Trace effect: ACT's drifted routing is MORE cacheable** (affinity policy:
   0.7952 on ACT trace vs 0.7369 on affinity trace) — mode-collapse onto a
   smaller expert set, same phenomenon as 2-GPU Q3-B. ACT's placement wastes
   more than its friendlier trace gains.
4. **The model is NOT the problem**: evaluated over whole trajectories with
   cache dynamics, the model's predicted ms/tok ranks affinity above the solver
   on BOTH traces (48.2 vs 51.4; 55.6 vs 73.5). Each per-layer solve is optimal
   for its layer; the SEQUENCE of solves loses — the definitive quantification
   of TD-LOADER-ROUTING-CROSSTOKEN myopia.
5. **Upside**: pooled-LRU-aware miss placement on ACT's trace projects to
   0.795 hit / 48 ms-of-fetch — ABOVE the current affinity champion (0.737).
   The cross-token reuse reward (place_cons) is worth more than the ACT-vs-
   affinity gap itself.

---

# 2026-07-18 addendum: reuse reward landed — hit parity; residual wall = trace shape

The place_cons reuse reward (sim-swept here: `reuse:inv:2000:2400` beat/matched
the affinity policy on both traces) shipped to the engine. Live keeper:
- **Hit gap CLOSED**: reuse-ACT 0.7373 vs affinity 0.7370 (was 0.719).
- 3×-interleaved walls: reuse-ACT 7.902 vs affinity 8.785 avg — the residual is
  NOT placement: per-miss-layer prefin p50 814 vs 806 µs (parity), fetch
  distribution [16.7,18.3,32.4,32.7]% vs [18.5,18.7,31.3,31.4]% (5080-heavy in
  both), stacking 19.2% vs 21.3%, DMA/xfer 799 vs 847 µs (reuse cheaper).
- Decomposition (perf-trace prefin, dump-joined): **~8 ms/tok = trace
  clustering** — reuse-ACT's drift equilibrium spreads equal misses over 419
  more layers/run, each paying a fetch wall (TD-LOADER-TRACE-CLUSTERING);
  **~3.4 ms/tok = loader machinery on zero-miss layers** (211 µs vs 2 µs/layer)
  — now fast-pathed (all-pinned layers skip the solve; behavior-neutral,
  bit-identical trajectory).
- Falsified along the way: TP-rank fetch bias (reuse is MORE 5080-heavy than
  affinity) and same-layer stacking (affinity stacks more) as gap causes.
Verdict: affinity remains the B=1 keeper champion by trace luck + zero
machinery; reuse-ACT is at placement parity, engine-side, orchestrator-
independent, and the path that generalizes to batch/prefill makespan work.

---

# 2026-07-18 addendum 2: canonical EP combine collapses the arms to ONE trajectory

`LAYERSTORM_DETERMINISTIC_EP_COMBINE=1` (+bf16 payload) on the EP=4 XTP keeper:
reuse-ACT and affinity dumps are **7500/7500 record-identical** — placement is
numerically inert on this topology (the 2-GPU-era INV-DRIFT-EPCOMBINE result
holds through the XTP per-slot fold). TD-LOADER-TRACE-CLUSTERING is therefore
an ARTIFACT of running with the gate off, not physics.

Shared-trace live/sim matrix (same trajectory for every row):
- affinity: engine 0.7233 == sim 0.7233 (EXACT — the orchestrator's bookkeeping
  IS the decision state, closed loop self-consistent)
- sim reuse policy (designed): **0.7244** — matches/beats affinity, as swept
- engine reuse-ACT: **0.6766** (choices replayed on clean LRU: 0.6848)
- solver declined a resident replica in only 2/19401 misses — the solve is
  faithful to its inputs; the ~4pp engine-vs-designed-policy gap is CLOSED-LOOP
  STATE DIVERGENCE (board-age/eviction fidelity vs the idealized pooled LRU the
  policy was developed on; engine-eviction-vs-clean-LRU itself is only −0.8pp).
  Filed TD-LOADER-REUSE-ENGINE-FIDELITY.
- canonical-combine wall cost at B=1/topk=8: affinity 8.042 vs ~8.8 gate-off
  (~9% — the per-slot payload is topk× the legacy combine) ⇒ dev/diagnostics
  gate at decode, not a production default.
