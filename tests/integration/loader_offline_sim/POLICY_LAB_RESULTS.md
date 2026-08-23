# Policy lab (sim-only saga, 2026-07-18): max out predicted tok/s — results

Instrument: `policy_lab.py` — decoupled eviction/placement policy zoo + Belady
oracles + online-trained models, replayed on the three committed trajectories
(primary: `traj_4gpu_canonical.jsonl.gz` — routing is policy-invariant under
deterministic_ep_combine, so sim results transfer). Metric: hit rate + predicted
fetch-ms/tok via the calibration transfer model. CPU only; no GPU runs.

## Headline table (canonical trace)

| policy | hit | pred ms/tok |
|---|---|---|
| LRU + affinity-place (today's champion) | 0.7234 | 57.7 |
| landed reuse-place (LRU eviction) | 0.7245 | 54.3 |
| **freq(d=0.1,w=500) eviction + reuse-place** | **0.7323** | **53.5** |
| freq + ES-trained linear place | 0.7319 | 52.9 |
| oracle: binary "recurs ≤ 8 tok" | 0.7907 | 48.9 |
| oracle: binary "recurs ≤ 16 tok" | 0.8150 | 45.0 |
| **Belady bound (partitioned == pooled)** | **0.8240** | **43.4** |

Cross-trace (affinity / ACT-noevict traces): the freq+reuse combo holds
(0.7414/52.1 vs champion 0.7371/55.7; 0.7945/45.4 vs 0.7951/48.2) — hit +0.4-0.9pp
or tie, pred −6..−9% everywhere. Belady 0.826-0.858.

## Findings

1. **~10pp of hit-rate headroom exists** (0.7234 → 0.8240) and partitioning is
   free at the optimum (pooled Belady == partitioned 4-way).
2. **The bridge is a BINARY signal**: protecting exactly the experts that recur
   within ~16 tokens recovers 0.815 of Belady's 0.825. Short horizons (≤4 tok)
   add nothing — LRU already holds those.
3. **Occurrence statistics cannot cross the bridge**: base recurrence rate is
   0.824; trailing-window presence predicts it at 0.877 precision (barely above
   base), an online logistic on count/gap features LOSES to LRU (0.7064 — it
   protects everything), LRU-K loses (0.7081), EWMA-of-gaps is structurally
   un-heapable (overdue-decay) and inverted variants score 0.24. The one-shot
   18% class is what must be identified, and past occurrence doesn't mark it.
   MISSING FEATURES worth capturing: per-expert GATE WEIGHT / top-K rank (the
   dump does not record them today — extend LS_LOADER_SHADOW_DUMP), and EPM
   (Phase-29) hidden-state recurrence prediction, whose job this exactly is.
4. **Practical online win, engine-mappable now**: decayed-frequency eviction
   bonus (raw = recency + w·freq, d=0.1, w=500 — the EvictScoreBoard's external
   score API accepts exactly this) + the landed reuse-place term:
   **+0.9pp hit, −7..−8% predicted fetch time** over the affinity champion,
   robust across all three traces.
5. ES-trained linear placement (7 causal features) converges onto the
   hand-designed reuse-place (52.9 vs 53.5 pred; interpretable weights: spread
   +1.26, free-slot −1.48, cheap-link +0.72, old-victim −0.19) — the placement
   design is at its feature-set optimum; further placement gains need new
   information, not new tuning.

## Engine-transfer caveat

TD-LOADER-REUSE-ENGINE-FIDELITY still applies: the engine's board-state copy of
these signals must be faithful, or closed-loop compounding eats the margin
(measured −4pp on the reuse term before). Fidelity fix first, then the freq
bonus rides the same (now-trustworthy) state.
