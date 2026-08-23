# Loader offline-sim fixtures & reference

Self-contained inputs for the CPU loader offline-simulation integration test.
The SAME fixtures (`routed_trace.csv` workload + `gpu_loader_calibration_5090x2.json`)
also drive `decay_drift_probe_test` (fab1-evict-drift-probe), which diffs the
decay-on vs decay-off decision streams on the single fixed trace — see RESULTS.md
§DRIFT-PROBE. Oracle CSVs remain ASSERTION/DEBUG-only (never a decision input).

**CANONICAL-TRAJECTORY REFRESH (fab1-sim-fixture-refresh, 2026-06-30, eb3fc650) —
the fixture is now the post-all-fixes canonical trajectory.** Re-captured from the
engine with ALL THREE EP fixes landed (placement-invariant canonical combine +
force-ON dedup + zero-resident per-slot zero-fix) + det-reduce default-ON. The OLD
0.40 epoch (baseline 0.4002) is RETIRED: its trace predated the canonical combine,
when placement perturbed the bf16 EP-combine into a routing butterfly (ACT+decay
0.6291). **With the canonical combine, placement is routing-INERT — all 5 scenarios
share the ONE trajectory `34b429e3`** (verified: new `routed_trace.csv` == canonical
`base.drift` routing, 0/5800 mismatches; baseline==plain-ACT==ACT+decay md5). The
variants now differ ONLY on the EVICTION axis. The open-loop sim reproduces ALL FIVE
to <0.001 (baseline 0.6663 EXACT; recency 0.6548 ≫ hash 0.2573; ACT+decay 0.6550 ==
plain-ACT — butterfly GONE). New affine `C = 85.948 ms/token` (baseline 6.945 tok/s).
See RESULTS.md §CANONICAL-TRAJECTORY REFRESH. The 5-variant table + "Recorded-engine
reference" below are SUPERSEDED (0.40 epoch), kept for history.

**UPGRADE (fab1-offline-sim-engine):** the test no longer runs a local eviction
model. It drives the **REAL engine eviction** — a CPU `memory::ExpertCache`
(`NullBackends`, 460 stable slots/GPU), the `gpu_loader::EvictScoreBoard` as the
cache's `ResidencyListener`, and the SHARED `gpu_loader::apply_far_evictions`
victim selector (the SAME function the daemon calls) — plus the real
`LoaderSolver` placement + T(j) cost model. Residency/scoring/eviction all come
from the engine classes ⇒ sim hit-rate == engine hit-rate by construction.

**EP-COMBINE DEDUP (fab1-sim-dedup):** the sim also MIRRORS the engine's force-ON
cross-GPU EP-combine dedup (TD-MOE-EP-COMBINE-RESIDUAL fix, INV-MOE-EP-DISJOINT).
The sim models the EP dimension — per-GPU residency + an ACT reroute that CAN
co-reside a routed expert on >1 GPU — so the duplicate the engine dedups is
representable here. `offline_sim::ep_combine_owners()` REUSES the pure
`daemon::dedup_ep_residency()` header to pick each routed expert's compute owner =
lowest-rank holder, which the T(j) cost model then single-counts exactly as the
deduped engine combine does. Pure accounting (the cache is NOT mutated ⇒ hit-rate
unchanged); a byte-identical no-op on the disjoint baseline. On the captured fixed
trace ACT mints ~0 co-resident routed duplicates so the correction is empirically
negligible THERE; the double-count regime is exercised by the synthetic
`EpCombineOwners` tests. See RESULTS.md §EP-COMBINE DEDUP FIDELITY.

**FIDELITY FIX (fab1-offline-sim-fidelity).** All 5 variants now replay the
SINGLE baseline trace `routed_trace.csv`, varying ONLY the algorithm (ACT /
board / decay / fallback) — the algo-not-data mandate. The prior epoch drove each
variant off its OWN policy-drifted oracle trace (`oracle_act.csv` /
`oracle_decay.csv`), which manufactured a SPURIOUS decay inversion: those traces
have wildly different intrinsic cacheability (clean-LRU ceiling `oracle_act`
0.7629 vs `oracle_decay` 0.3931), so the table compared apples to oranges. On the
single trace the inversion vanishes and **plain-ACT NEW reproduces the keeper
near-exactly** (sim 0.3906 vs keeper 0.392). The oracle CSVs are now read ONLY
for the recorded-engine hit@j reference; they never drive a variant.

## Assets
- `routed_trace.csv` — the routed-expert workload, captured from the keeper
  `RoutedExpertLruFullFitDirectTest.HundredTokenDecodeFetchAndRun_FullFit_EP2`
  (shadow mode), under the BASELINE policy. **RE-CAPTURED 2026-06-30 (fab1-sim-
  fixture-refresh) from the engine with ALL THREE EP fixes (canonical combine) +
  det-reduce — the canonical `34b429e3` trajectory.** Verified bit-identical routing
  to the canonical `base.drift` (0/5800 layer-visit mismatches). Regenerated via
  `LS_LOADER_SHADOW_DUMP` JSONL → `dump_to_trace.py` (token = solve order, layer_idx
  3..60/token). Columns: `token,layer,slot,expert_idx,bank`.
  - 100 tokens × 58 MoE layers × K=8 routed = 46400 rows.
  - `layer` ∈ [3,60] (first 3 layers are dense, no MoE). `slot` ∈ [0,8) is the
    routed rank within the layer. `bank` = the expert's host NUMA source node.
  - **THE single workload for all 5 variants** (algo-not-data). FP routing-drift
    is treated as negligible per the fidelity mandate.
- `oracle_baseline.csv` — 9-col (`…,oj,j,cached0,cached1`); cols 1-5 are IDENTICAL
  to `routed_trace.csv`. The `cached0,cached1` residency columns assert the
  baseline variant's recomputed solve-time residency bit-for-bit (0 divergences).
- `oracle_act.csv`, `oracle_decay.csv` — **RE-CAPTURED 2026-06-30** from the ACT /
  ACT+decay scenarios on the SAME canonical `34b429e3` trajectory (routing cols 1-5
  are now IDENTICAL to `routed_trace.csv` — placement is routing-inert; only the
  residency cols `oj,j,cached0,cached1` differ, the eviction axis). Read only for the
  recorded-engine hit@j reference line (act 0.6547 / decay 0.6547). **Not a sim driver.**
- `gpu_loader_calibration_5090x2.json` — the symmetric 2×5090 calibration (load
  via `gpu_loader::from_json_string()`). M=2 devices, B=4 banks. dev0 home=node2,
  dev1 home=node3; home-node H2D ≈ 56 GB/s (symmetric).

## System config (for the cache simulator)
- tp = 2 (M = 2 GPUs). `e%tp` = `expert_idx % 2`.
- Expert cache: **460 stable slots per GPU**, LRU eviction, FullFit (host arena
  holds all experts; a VRAM miss = one H2D transfer, always reachable).
- 256 experts/layer, K=8 routed/layer, expert_bytes = 24,772,992 (24.77 MB).

## Final 5-variant table — [SUPERSEDED 0.40 epoch; see CANONICAL-TRAJECTORY REFRESH at top]
Refreshed baseline trace; keeper hits BIT-STABLE across 3 idle-checked full-pool
runs (zero hit-rate spread). tok/s = 3-run mean; sim tok/s anchored to baseline
5.275 (`C = 96.217 ms/token`).

| variant           | sim hit | keeper hit | hit err | keeper tok/s | sim tok/s | note |
|-------------------|---------|------------|---------|--------------|-----------|------|
| baseline          | 0.4002  | 0.4002     | 0.0000  | 5.275        | 5.275     | **EXACT**, 0 oracle divergences |
| plain-ACT NEW     | 0.3900  | 0.3926     | −0.0026 | 5.463        | 5.637     | open-loop predicts it CLOSE |
| plain-ACT legacy  | 0.2341  | 0.2835     | −0.0494 | 5.260        | 5.211     | hash-order; trace-sensitive |
| ACT+decay NEW     | 0.3900  | 0.6291     | −0.2391 | 7.078        | 5.598     | **butterfly** — open-loop CANNOT predict |
| ACT+decay legacy  | 0.2342  | 0.2808     | −0.0466 | 5.230        | 5.215     | hash-order; trace-sensitive |

clean-LRU ceiling (refreshed trace) = 0.4002. Cross-GPU duplicate residency ≈
0.004% for every ACT variant; ≤ 2 duplicate inserts over 46400 lookups.

**Determinism reframe:** the keeper's 0.6291 is NOT sampling noise — with det-reduce
ON it is bit-stable ×3 (atomic-epoch spread gone). It is a DETERMINISTIC closed-loop
placement-butterfly the open-loop fixed-trace replay cannot see. Superseded atomic
values: baseline 0.4015 / plain-ACT 0.392 / ACT+decay 0.629 @ 5.249. See RESULTS.md
§DET-REDUCE-REASSESS for the full hit + tok/s error analysis and verdicts.

## Verdict — it was a TEST-DATA bug; the engine mechanism is FAITHFUL
1. **Spurious inversion = per-policy-trace cacheability**, not the algorithm. Fixed
   by replaying the single baseline trace. plain-ACT NEW then matches the keeper.
2. **LS_EVICT_DECAY is a NO-OP on a fixed workload.** `decay_all` multiplies every
   raw by the same factor; with no cross-GPU duplicates (rank 0 ⇒ eff == raw) a
   uniform scale is ORDER-PRESERVING per GPU ⇒ identical victim selection ⇒
   identical residency. Sim: ACT+decay NEW 0.3905 ≈ plain-ACT NEW 0.3906.
3. **Hypothesis (placement mints duplicates) DISPROVEN.** The `LoaderSolver` routes
   a miss to a GPU where the expert is already cached (free transfer), so cross-GPU
   duplicates essentially never co-reside (dupRes% ≈ 0.004). The keeper's
   duplicate/decay regime is therefore **routing-drift-generated**, not placement-
   generated. For the decay metric, drift is NOT negligible: the keeper's 0.629 is
   an emergent property of the closed-loop autoregressive workload and cannot be
   reproduced from a fixed trace by construction.
4. **Recency-magnitude (TD-EVICT-RECENCY-MAGNITUDE).** The always-on recency clock
   reaches ~5800 over the run, so the additive duplicate penalty `max(0, raw −
   rank·0.9)` is noise — even WHERE a duplicate exists it is not preferentially
   evicted. So decay cannot bite via the duplicate path either. The DEBUG.md claim
   that "decay keeps raws O(base)" is false: touches re-stamp raw to the growing
   clock, so 0.98×/token never brings raws near 0.9 in a 100-token run.

## Recorded-engine reference — SUPERSEDED (hash-order epoch; == oracle fixtures' hit@j)
| method        | tok/s | hit_rate | fetched/token |
|---------------|-------|----------|---------------|
| baseline e%tp | 5.313 | 0.4016   | 277           |
| ACT no-decay  | 5.452 | 0.3091   | 320           |
| ACT + decay   | 5.034 | 0.2349   | 355           |

These are the older engine epoch that produced `oracle_act/decay.csv`. The current
keeper A/B (recency / `cheapest_keys` path) is baseline 0.4015 · plain-ACT NEW
0.392 · plain-ACT legacy 0.282 · ACT+decay NEW 0.629 · ACT+decay legacy 0.278,
baseline at 5.249 tok/s (the affine anchor).

## Known mechanism (what the sim reproduces)
- ACT wins by **fetch load-balancing**: it balances per-GPU per-layer miss count
  (≈45.7→21.8 ms/tok), which outweighs its near-equal hit-rate. hit-rate is not the
  lever (sim T_pred drops 93→81 ms with reroute; hit-rate ~flat).
- The T(j) cost model and real tok/s are known to DIVERGE (model under-predicts
  makespan ~1.48× and omits host launch tax / attention / handoff). The test
  measures predicted-tok/s anyway to quantify the divergence.
