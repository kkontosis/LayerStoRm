# Dethrone-affinity ideas ledger (keeper52 saga, 2026-07-18)

GOAL: REEF (the I8-ACT champion stack) must beat keeper-affinity on WALLS, not
just decision quality. Standings when filed: gate-on same-trace REEF hit 0.7340
> affinity 0.7233 (decisions won); gate-off walls affinity 8.82 > REEF 7.47
(trace lottery + unpaid wall bill). Keeper now defaults the canonical combine
ON (setenv in SetUp) — all future keeper A/Bs are same-trace by construction.

## Measured-now facts the ideas build on
- M2 exposed-wall model (tools/loader_xray/exposed_model.py): transfer_wall R²
  0.712→0.824, total 0.2→0.735 (config-scoped cpw/b0; cross-config total does
  NOT generalize — refit per deployment). cpw=[2.06,2.34,1.87,1.57]: a 5090/TP
  stall costs the token wall ~30-50% more than a 5080 stall. Overlap credit
  ~28-66 µs per resident expert computed. Sim absolute scale now honest
  (83 ms/tok predicted vs ~85 measured for the champion arm).
- Belady ceiling 0.824 hit; binary 16-tok recurrence bridge recovers 0.815;
  occurrence statistics provably cannot cross (POLICY_LAB_RESULTS.md).

## Ideas (ranked)
1. **Pay REEF's wall bill** — gate-on trio (affinity / bare-ACT / REEF, no
   dumps): same trace ⇒ wall deltas = machinery + placement pattern. Suspects:
   touch_existing_all rerank per access, place-term solve paths, per-layer miss
   clustering of the drawn trace. Fix what the split names.
2. **cpw-weighted placement** — feed M2's critical-path weights into the solver
   place term (place cost += (cpw_j−1)·expected_exposed_j): steers misses away
   from TP ranks ONLY when their stall would be exposed. Supersedes the naive
   TP-penalty probe (which was hit-neutral but untargeted). Sim first (M2 pred
   now in trajectory_sim via set_exposed / --exposed).
3. **Exposed-aware placement objective in C++** — port M2's per-device exposed
   form into LoaderSolver's makespan term (overlap credit from hits_j, cpw
   scale); constants extend LoaderConstants v3; trainer gains cpw/oc coefficients
   (config-scoped). The solver then minimizes what the wall actually pays.
4. **Recurrence predictor** (the +9pp): extend LS_LOADER_SHADOW_DUMP with gate
   weight + top-K rank per expert (one-line dump change) → retrain the one-shot
   classifier with weight features; if AUC clears, wire as board protection.
   EPM Phase-29 hiddens are the endgame form.
5. **Orchestrator-side REEF loop closure** — affinity is immune to state-fidelity
   drift because its bookkeeping IS its decision state. A python-orchestrator
   REEF (decide from the same LRU the orchestrator maintains, send explicit
   gpu_idx) removes the board-fidelity class entirely.
6. **Gate-on production** — canonical combine ~9% tax; REEF wins on merit under
   it. Viable if (1)+(2) recover more than the tax on the shared trace.
7. **Freq/reuse retune on the shared trace** — w/τ/d sweeps are now clean
   single-keeper-run experiments (hit is deterministic per config); cheap grid.

## 2026-07-18 UPDATE: wall-bill trio verdict + M2 port HALTED (user decision)

Trio (gate-on, same trace, round 1; round 2 confirms pending):
| arm | hit | avg tok/s |
|---|---|---|
| affinity | 0.7233 | 8.550 |
| bare-ACT (no reuse/freq) | 0.6637 | 7.554 |
| REEF | 0.7340 | 7.490 |

**FINDING: the wall bill is a FIXED ACT-path cost (~16 ms/tok), not decisions.**
REEF carries +7pp hit over bare-ACT at the SAME wall — hit does not convert to
wall on the ACT path. The ~1.05 tok/s gap to affinity is the ACT execution
path itself (suspects: reroute/fetch-by-j bookkeeping, eviction-map
maintenance under reroute, overlap-pass behavior when misses execute off their
orchestrator target). NEXT INSTRUMENT: perf-trace decomposition bare-ACT vs
affinity (same trace ⇒ clustering identical ⇒ any prefin/finalize delta is
pure path cost). Idea #1 is now THE gate: no decision-quality work converts
until this bill is paid.

**M2 C++ port HALTED (user, 2026-07-18)** — working-tree edits reverted. The
B=1-scoped form is not an acceptable foundation; M2v2 requirements before any
port:
- keep the batch-step compute term (a·c + b·⌈c/P⌉) alongside the exposed
  transfer wall — do not let the credit absorb compute (wrong at B>1);
- SATURATING overlap credit (e.g. o0 + oc·min(hits, h_sat) or 1−exp form),
  not linear-in-hits;
- per-device exposure valid at arbitrary miss counts (prefill shapes);
- fit methodology must exist for B>1 first: batched capture harness (canonical
  combine is single-shot-only; chunked+extras fails loud) — build the harness,
  refit, THEN port;
- pinned-tier-only guard (B&B bound validity) stays mandatory in any port.

## 2026-07-18 UPDATE 2: force-identity control SPLITS the bill (user's experiment)

LS_LOADER_FORCE_IDENTITY=1 + affinity orchestrator (affinity's exact placement,
full ACT machinery, zero reroutes; gate-on): **7.648 tok/s, hit 0.6960** (2×
identical). Identical placement lost −2.7pp hit vs plain affinity (0.7233) ⇒
**the acting path's EVICTION plumbing selects worse victims than the plain
path even when no decision changes**. Bill decomposition: ~10 ms/tok eviction-
path quality + ~5 ms/tok true machinery. This also resolves the trio paradox:
REEF's +7pp placement gain swims against the acting path's eviction current.
NEW TOP SUSPECTS (idea #1 refined): board/cheapest_keys-driven victim
selection vs the production recency map; TD-FAR-EVICT-REROUTE eviction-map
machinery active even with zero reroutes. Next: diff the two eviction paths'
victim streams on the canonical trace (both sides log evictions), or unit-level
A/B of ExpertCache eviction with/without the acting-path plumbing.

## 2026-07-18 UPDATE 3: eviction-score fix RESOLVED — accounting closed

Probes (gate-on): force-identity + LS_LOADER_FREQ_W=0 → hit **0.7233 = plain
affinity EXACTLY** (16603 misses, 166 fetch/tok) at 8.369 tok/s. VERDICT: with
freq off, the board IS a bit-faithful pooled LRU — the entire −2.7pp "eviction
defect" was the freq bonus distorting recency (w·f up to ~630 clock units ≈ 8
tokens of artificial youth) under a FOREIGN placement policy. No architecture
fix needed. REEF freq sweep: w 0/7/15/60/125/250 → hit .7230/.7263/.7292/
**.7340**/.7071/.6164 — clean inverted-U, default w=60 IS the peak (matches
sim shape). CONSTRAINT: freq bonus is placement-policy-coupled — set 0 whenever
the board serves eviction for non-solver placement (diagnostic arms).

**Closed wall accounting (gate-on):**
- affinity 8.546 = fid_w0 8.369 + 0.18 machinery (3.1 ms/tok — books balance)
- REEF-w0 7.988 vs fid_w0 8.369 at EQUAL hit/eviction ⇒ **solver placement
  PATTERN costs ~0.38 tok/s of wall at equal hit** — now the dominant lever.
- REEF 8.11 (hit .7340) vs affinity 8.546: gap 0.44 = ~0.18 machinery
  + ~0.26 net placement-pattern (partially offset by +1.1pp hit).
NEXT: the pattern lever is exposed-wall/critical-path placement = the M2v2
line (cpw-aware placement was sim-validated at −4% pred wall, equal hit).

## 2026-07-18 UPDATE 5 (P-25): REEF EXPRESSES AFFINITY — existence proof GREEN

`LS_LOADER_PLACE_AFFINITY=1 LS_LOADER_REUSE_W=0 LS_LOADER_FREQ_W=0` (implies
full `LS_LOADER_ABLATE=xfer,compute,bank,recon`): the exact solver, given the
synthetic count/age objective (loader_affinity_place.{h,cpp}) over ZEROED cost
constants, reproduces the keeper affinity router THROUGH the loader path —
**hit 0.7233 (fingerprint-exact print), 16599 misses vs plain affinity's
16603** (99.99% decision agreement; residual −4 = near-tie tail). Mechanism
notes (each earned by a measured divergence):
- count balance = convex `evict_cum[j][n] = BIG·n(n−1)/2` (place is linear,
  cannot express it); age tie = `place[i][j] = MED·(agerank_j+1)` — the +1
  keeps a fetch on the oldest device strictly costlier than riding a replica
  (first property-test failure was this tie);
- pairing (which miss → which chosen device at ≥2/device) is a solver
  CO-OPTIMUM — canonicalized post-solve to the router's per-level round-robin;
- the board's per-layer grouped clock CANNOT order same-layer victims across
  devices (witnessed first E0 divergence: ticks 34 vs 37, one layer-visit
  apart → board tie → wrong device; cost −4.5pp at 0.6881). Fix: dispatcher
  fine-tick mirror stamped in the keeper model's application order (per-GPU
  grouped, hits before misses, routed order within) + needed_now exclusion.
- Property test: tests/unit/loader_affinity_place_test.cpp (2000 random
  configs == reference greedy; warmup round-robin; free-slot precedence).
- REEF control with the mechanism landed but OFF: hit 0.7340 unchanged ✓.
Wall: E0 7.575 avg first run — BELOW the saga board-LRU 8.369 at equal
decisions; needs mach-prof + same-day interleaved anchors before reading.

## 2026-07-18 UPDATE 8: AFFINITY DETHRONED — KEEPER52_REEF_ORCH (decision-3 scaffold)

The test-as-orchestrator REEF scaffold (KEEPER52_REEF_ORCH=1: real LoaderSolver
+ real EvictScoreBoard test-side, explicit gpu_idx + 13c-2.0 victim map, engine
loader OFF ⇒ zero reroutes, bank term ON per the user decision):
| arm (same day, interleaved, gate-on, post-LASTPASSER epoch) | hit | tok/s |
|---|---|---|
| **REEF-ORCH (new champion)** | **0.7327 (×2 bit-stable)** | **8.815 / 8.862** |
| affinity plain (interleaved anchor) | 0.7233 | 8.541 |
| board-LRU | 0.7233 | 8.424 |
| engine-REEF (post-rebase) | 0.7328 | 8.172 / 8.035 |

**+0.31 tok/s over affinity at +0.94pp hit — the ledger's GOAL line is met**:
REEF beats keeper-affinity on WALLS, on merit, with the full I8 objective
(bank included). The winning combination = REEF decisions (worth the hit) +
the zero-reroute dispatch shape (worth ~0.65 vs engine-REEF: hits pre-targeted
by the orchestrator arrive was_cached, no replica-hit reroute machinery).
Decision-parity check: test-side hit 0.7327 vs engine-side 0.7328 — the two
independent implementations of the REEF decision state agree to ~1 flip.
Env: KEEPER52_REEF_ORCH=1 LS_LOADER_CALIB=gpu_loader_calibration_4gpu_hbm.json
(engine LS_LOADER_SHADOW forced 0 by the harness). This is the A-compatible
scaffold; production form = the same loop closure in the Python orchestrator
(idea #5 / P-18 line).

## 2026-07-18 UPDATE 9: freq/reuse regrid under the NEW epoch (REEF-ORCH arm) — defaults CONFIRMED

12-point grid through KEEPER52_REEF_ORCH (knobs plumbed test-side, same env
names/defaults as the dispatcher; control 60/2000/300 reproduced fingerprint
0.7327 = plumbing decision-neutral). Freq sweep (reuse 2000/300):
w 0/15/30/45/60/75/90/125 → hit .7221/.7280/.7319/.7331/**.7327**/.7293/
.7241/.7063, walls 8.768→8.925→8.594 — the inverted-U survives the LASTPASSER
rebase with the SAME peak region; wall tracks hit at ~0.15 tok/s per pp.
Reuse/tau corners at fw=60: rw 1000/4000 → .7313/.7339; τ 150/600 →
.7325/.7320; walls all 8.919-8.935 = FLAT (same-config spread today is ~0.1,
so no corner separates). rw=4000's +0.12pp hit predicts ~+0.02 wall —
inside noise, not worth the interleave burn. **VERDICT: freq_w=60,
reuse_w=2000, τ=300 sit on the plateau optimum under the new epoch — landed
defaults stand unchanged.**

## 2026-07-18 UPDATE 10: trained-calib probe through REEF-ORCH — untrained calib stays

One-run probe (P-26 QA): KEEPER52_REEF_ORCH with the I8-trainer-corrected
`gpu_loader_calibration_4gpu_hbm_trained.json` → **8.703 @ hit 0.7307** vs the
champion's 8.815-8.935 @ 0.7327 on the UNTRAINED hbm calib. The trainer's
corrections (fit to reduce wall-PREDICTION error) are decision-NEGATIVE here
(−0.2pp hit) — INV-LOADER-OBJECTIVE-MYOPIC yet again: prediction accuracy and
placement quality are different objectives. Canonical champion arm keeps
`gpu_loader_calibration_4gpu_hbm.json`; the trained file remains a prediction
asset (x-ray/trainer line), not a decision asset.

## 2026-07-18 UPDATE 11: P-26 confirm loop exercised end-to-end — sim winner REJECTED

Full pipeline demonstration on the driver's own output: deploy_fit --only
policy emitted fw=120/rw=4000/tau=150 (sim hit 0.7294, ranked #1 of 54);
single-run keeper confirm through REEF-ORCH → **hit 0.7083 @ 8.623** vs the
defaults' 0.7327 @ 8.8-8.9 — a −2.4pp keeper REGRESSION, fingerprint
backfilled and the artifact archived as committed evidence
(assets/policy_params_deployfit_fw120_rejected.json). QUANTIFIED LESSON: the
offline sim's freq model over-rewards protection (sim ordering INVERTS the
keeper's above w≈60); "sim ranks, keeper decides" is now a measured rule, not
a caution. Defaults 60/2000/300 remain the confirmed optimum.

## 2026-07-19 UPDATE 12: sim freq model FIXED — engine units, keeper ordering reproduced

Root cause of UPDATE 11's inversion: TWO clock-domain bugs in the
engine-faithful pair. make_sc_freq scored `t + w·f` with t = the PER-TOUCH
counter, while the engine board stamps `layer_clock + w·f` with the clock
advancing once per LAYER-VISIT (~topk≈8 touches) — the same w bought ~8×
less protection in the sim, so its grid top (fw=120) mapped to engine ≈15 and
the sim only ever saw the upslope. make_place_reuse had the matching bug
(age/τ in touch units: engine τ=300 layer-visits ran as sim τ≈37). FIX: both
now use ridx (the record index == the board's layer-visit clock, grouped-LRU
semantics); the one t-domain zoo pairing switched to make_sc_freq(w=0) (the
ridx-LRU == the board at FREQ_W=0).
VALIDATION at the keeper's 8 regrid points (rw2000/τ300): sim now reproduces
the inverted-U — fw 0/15/30/45/60/75/90/125 → .7244/.7287/.7317/.7322/
**.7328**/.7323/.7288/.7188 vs keeper .7221/.7280/.7319/.7331/.7327/.7293/
.7241/.7063 — peak position agrees (45-60), absolute error ≤1pp everywhere
and ~0 at the peak (sim .7328 vs keeper .7327). Full 54-point grid: top-7 all
fw=45-60 (plateau, spread .7326-.7331) == the keeper regrid verdict. Residual
model error: the sim downslope is flatter (75/90 over-rewarded ~+0.3-0.5pp) —
peak selection robust, but keep "sim ranks, keeper decides" for shoulder
calls. Historical t-domain zoo freq rows (POLICY_LAB_RESULTS.md) predate the
fix and are not reproducible with the new units.

## 2026-07-19 UPDATE 13: P-26 pipeline GREEN end-to-end (fixed sim)

Full chain on a FRESH capture: engine-REEF keeper w/ dump+trace (fingerprint
0.7328 = canonical, dump-neutral) → deploy_fit ALL stages in one invocation →
{loader_constants_trained.json, exposed_params_v2.json, policy_params.json};
policy winner 60/0.1/2000/τ150 (sim 0.7331) → single-run keeper confirm →
**hit 0.7325 @ 8.756 — EXACTLY the regrid's independently-measured τ150
fingerprint**, backfilled + archived
(assets/policy_params_deployfit_e2e_confirmed.json). Verdict: plateau point
tied with the defaults (0.7327 @ 8.8-8.9) — defaults remain; the pipeline now
ranks correctly (contrast UPDATE 11's pre-fix fw=120 rejection) and its
sim-hit predictions land within ~0.5pp of keeper truth on plateau points.
Constants/M2 artifacts are prediction assets per deployment (the trained
calib remains decision-negative for placement, UPDATE 10 — do not point
LS_LOADER_CALIB at it for decision arms).

## 2026-07-18 USER DECISIONS (P-25 QA round)
- **Bank term stays DEFAULT ON, generally on** — REEF−bank (8.13 @ 0.7324) stays a
  recorded diagnostic datum, NOT the default; `LS_LOADER_ABLATE=bank` remains an arm env.
- **LASTPASSER goes DEFAULT ON, generally on** — strict-improving leaf + re-land
  seeding/symmetry (26b36fa3 recipes); accept the ONE-TIME fingerprint re-baseline
  (new canonical REEF fingerprint to be recorded below when landed).
- **Reroute-hit path: long-term arch = A (orchestrator-side loop closure), stay
  A-compatible.** Start by scaffolding the orchestrator role INTO keeper52_test itself
  (test-side REEF: pure LoaderSolver + model residency/board, explicit gpu_idx targets,
  zero reroutes) — NOT into the engine. Revisit B (or an A/B hybrid) only if the test
  scaffold shows it's time-critical.

## 2026-07-18 UPDATE 6 (P-25): walk-back VERDICT — bank term named, "pattern" reframed

Single-term ablations from R0 = REEF+FREQ_W=0 (same day, gate-on, no-graph
recipe, one keeper at a time; fingerprints bit-stable per config):
| arm | hit | tok/s | Δ vs R0 |
|---|---|---|---|
| affinity plain | 0.7233 | 8.515 | (anchor) |
| board-LRU (fid+w0) | 0.7233 | 8.424 | (anchor) |
| R0 − bank | 0.7233 | 8.080 | **+0.10** |
| R0 | 0.7230 | 7.980 | — |
| R0 − compute | 0.7224 | 7.904 | −0.08 |
| E0 (solver-expressed affinity) | 0.7233 | 7.743* | (*mach-prof on) |
| R0 − xfer | 0.7086 | 6.158 | −1.82 |

- **Bank egress = the named single-term loser at B=1** (+0.10 wall at EXACTLY
  equal hit 0.7233): its spreading fights the real fetch overlap. xfer is
  strongly load-bearing (NUMA steering, −1.82 without it); compute ≈ neutral.
- **CHAMPION CANDIDATE: REEF − bank** (`LS_LOADER_ABLATE=bank`, freq 60):
  8.146 / 8.121 interleaved vs adjacent REEF 8.059 / 7.946 — wins both pairs
  (≈ +0.15), hit 0.7324 (−0.16pp; freq was tuned WITH bank — idea #7 regrid
  under bank-off may recover it). Default flip = user decision (fingerprint
  0.7340 → 0.7324).
- **"0.26 pattern" REFRAMED**: E0 runs the affinity PATTERN through the same
  reroute-laden ACT path as REEF — machinery-corrected (solve-enumeration
  excess ~31 µs/layer measured) it lands ≈ R0−bank, NOT ≈ board-LRU. The
  board-LRU arm's remaining +0.34 over R0−bank is the ZERO-REROUTE dispatch
  path: with an e%tp orchestrator, **3.76 replica-hits/layer (47% of routed
  experts, 28189/60000 counted in the E0 x-ray)** execute via reroute instead
  of arriving was_cached at dispatch; board-LRU reroutes zero. So the saga's
  0.26 "pattern" ≈ 0.10 bank term + ~0.3 reroute-hit EXECUTION cost —
  machinery, not decision quality. At equal machinery the affinity pattern
  holds no measurable edge over REEF−bank.
- NEXT LEVERS, reranked: (1) the reroute-hit path — orchestrator-side loop
  closure (idea #5; P-18-adjacent) or cheapen the daemon replica-hit arrival;
  (2) TD-LOADER-SOLVER-LASTPASSER re-baseline (~4× solve; also collapses E0's
  enumeration excess — its count term has no residual bound); (3) freq regrid
  under bank-off; (4) bank-off default decision.

## 2026-07-18 UPDATE 4: M2v2 line DONE — ported, opt-in, B=1-neutral

Form (all B>1 reqs met): compute first-class, saturating credit (hsat fits
8–46), valid at any miss count, credit frozen from pinned hits (monotone ⇒
pinned-tier-only arming; legacy makespan floors zeroed under m2). Fit parity
with v1 (wall R² 0.821, total 0.735). Sim placement: −3.7% pred at equal hit.
C++ port: LS_LOADER_M2=<json>, 4 solver tests, fingerprint hit 0.7333.
**GPU verdict: wall-NEUTRAL at B=1** (8.110/8.048 vs REEF 8.104/8.117) — the
predicted pattern gain did not convert; the 0.26 pattern gap (fid_w0 8.369
proof) is NOT captured by M2v2 steering at B=1 miss counts (~2/layer, little
placement freedom after pins+reuse). Stays opt-in. Where it should pay:
B>1/prefill shapes (many misses/layer = real placement freedom) — blocked on
the B>1 capture harness + refit, the declared prerequisite.
