#!/usr/bin/env python3
"""Train the ONE trainable never-lose place_cons weight — w_numa — against the
RESIDENCY-HONEST objective (closed-loop black-box fit).

WHY ONLY w_numa (design decision 2026-08-10): a weight on a PREDICTION cannot be
fixed by training when the prediction is wrong. The residency term w_resid·recur
ZEROES on the recur=0 churners (b0_prev mis-marks them safe ⇒ measured −13.7% with
any trained w_resid), and freq/EMA is likewise a prediction. So residency (a CONSTANT
per-offload churn TAX, resid_tax) and hotness (w_hotness) are HEURISTICS held fixed
here, and only w_numa — a STATIC home-node FACT, not a prediction — is trained. The
constant churn tax is what makes the never-lose greedy DECLINE the mis-priced recur=0
churners (the honest floor); w_numa tunes the cross-node host-read penalty.

METHOD (INV-LOADER-OBJECTIVE-MYOPIC: policy weights are fit by closed-loop offline
sim, NEVER regression). We replay the committed trajectory through the residency-
HONEST sim (cpu_offload_bsweep.run_B_honest machinery: offloaded experts are EXCLUDED
from the GPU cache advanced token-by-token, so the residency-churn cost is GROUND
TRUTH via cache exclusion — NOT a modeled estimate). The offload DECISION uses the
ENGINE'S ACTUAL FREE SIGNALS (b0_prev prev-chunk union + M3 freq prior), the exact
choose_cpu_offload net-relief greedy, and the tunable weights. We grid the weights
(and the overlap-belief eta the solver DECIDES with) and rank by honest tps_nl.

The objective is HONEST-eta accounted: the champion regime overlap is the MEASURED
eta-cap (~0.0-0.3, MILESTONE_C), so a weight set that offloads a churner PAYS the
exposed host FFN + the forfeited-caching future misses. The best weights are the
ones whose priced net-relief goes negative for the churners ⇒ the greedy declines
them ⇒ break-even (never-lose made real) or a sliver.

Usage:
  python3 train_place_sum_weights.py --trace assets/traj_4gpu_canonical.jsonl.gz \
      --pin 800 --caps 64,64,64,64 --resid-window 1 --eta-cap 0.0 --B 16 --out /tmp/w.json
"""
import argparse
import json
import math
import sys
from collections import OrderedDict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cpu_offload_bsweep as sim  # validated honest cost model + helpers
from trajectory_sim import load_traj


def run_B_priced_engine(by_layer, ntok, B, resid_tax, w_resid, w_numa, w_hotness,
                        freq, fhot, home_of, cpu_node, eta_decision, eta_account, M=4,
                        resid_window=1, pinned=None, caps=None):
    """Residency-HONEST replay of the ENGINE never-lose greedy with the weighted-sum
    place_cons. Mirrors dispatch_loader.cpp::choose_cpu_offload:
      place_cost[i] = w_resid·(recur_b0prev·fetch) + w_numa·numa + w_hotness·(freq/fhot)
      offload order = descending net relief (fetch − place_cost)
      wall(k)       = gpu_makespan + exposed_host + Σ place_cost(offloaded prefix)
      k* = argmin wall (k=0 champion always in search ⇒ never-lose).
    eta_decision = the overlap the greedy BELIEVES (solver's exposed-host estimate);
    eta_account  = the REAL overlap the honest wall is charged at (MEASURED). Churn is
    ground truth: an offloaded expert is EXCLUDED from the token-by-token GPU cache."""
    if pinned is None:
        pinned = set()
    if caps is None:
        caps = sim.CAPS
    caches = [sim.Lru(caps[g]) for g in range(M)]
    nchunks = math.ceil(ntok / B)
    fetch_amort = sim.FETCH_US / sim.FETCH_CHANNELS
    prev_union = [set() for _ in range(sim.NLAYERS)]  # b0_prev: prev chunk's union/layer
    tot_all = tot_nl = 0.0
    tot_off = tot_recur_off = 0
    gpu_lookups = gpu_hits = 0
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        bsz = t1 - t0
        for layer in range(sim.NLAYERS):
            uni = {}
            for t in range(t0, t1):
                for e in by_layer[layer][t]:
                    uni[e] = 1
            U = len(uni)
            resident = {e for e in uni
                        if (layer, e) in pinned or (layer, e) in caches[e % M].od}
            miss = [e for e in uni if e not in resident]
            m = len(miss)
            prev = prev_union[layer]
            # ── engine place_cost per candidate miss ─────────────────────────
            cand = []  # (net_relief, place_cost, e)
            for e in miss:
                recur = 1.0 if e in prev else 0.0
                resid_baseline = recur * fetch_amort           # would-be-HIT next round
                node = home_of(e)
                numa_pen = 1.0 if (node >= 0 and node != cpu_node) else 0.0
                h = min(1.0, freq.get((layer, e), 0) / fhot) if fhot > 0 else 0.0
                # resid_tax + w_resid (HEURISTICS) + w_numa (TRAINED) + w_hotness (HEUR)
                place = max(0.0, resid_tax + w_resid * resid_baseline
                            + w_numa * numa_pen + w_hotness * h)
                cand.append((fetch_amort - place, place, e))
            # descending net relief (biggest net first) == engine pop-from-back
            cand.sort(key=lambda x: -x[0])
            # ── never-lose k-sweep on the honest wall (engine choose_cpu_offload) ─
            base_wall = sim.layer_wall_allgpu(U, m, bsz)
            best_wall, best_k = base_wall, 0
            cum_place = 0.0
            for k in range(1, m + 1):
                cum_place += cand[k - 1][1]
                gpu2 = max(sim.per_gpu_fetch(m - k, sim.DMA_CONTENTION),
                           sim.per_gpu_compute(U - k, bsz))
                cpu2 = sim.FOLD_US + k * sim.host_expert(bsz)
                exposed = max(0.0, cpu2 - eta_decision * gpu2)
                wall = gpu2 + exposed + cum_place
                if wall < best_wall - 1e-9:
                    best_wall, best_k = wall, k
            K = {cand[i][2] for i in range(best_k)}
            k = best_k
            tot_off += k
            tot_recur_off += sum(1 for e in K if e in prev)
            # ── honest accounting wall (charged at the MEASURED eta) ─────────
            m_gpu = m - k
            gpu2 = max(sim.per_gpu_fetch(m_gpu, sim.DMA_CONTENTION if k else 1.0),
                       sim.per_gpu_compute(U - k, bsz))
            if k:
                cpu2 = sim.FOLD_US + k * sim.host_expert(bsz)
                exposed = max(0.0, cpu2 - eta_account * gpu2)
                wall = gpu2 + exposed
            else:
                wall = base_wall
            tot_all += base_wall
            tot_nl += wall
            # ── honest token-by-token residency advance (churn = ground truth) ─
            for t in range(t0, t1):
                for e in by_layer[layer][t]:
                    key = (layer, e)
                    g = e % M
                    if key in pinned:
                        gpu_lookups += 1
                        gpu_hits += 1
                        continue
                    if e in K:
                        caches[g].evict(key)
                        continue
                    gpu_lookups += 1
                    if caches[g].hit(key):
                        gpu_hits += 1
                    else:
                        caches[g].insert(key)
            prev_union[layer] = set(uni.keys())
    fixed = sim.FIXED_TOK_US * ntok
    tps_all = ntok / ((tot_all + fixed) / 1e6)
    tps_nl = ntok / ((tot_nl + fixed) / 1e6)
    return {
        "off_total": tot_off,
        "recur_off": tot_recur_off,
        "recur_off_frac": tot_recur_off / max(tot_off, 1),
        "hit": gpu_hits / max(gpu_lookups, 1),
        "misses": gpu_lookups - gpu_hits,
        "tps_all": tps_all,
        "tps_nl": tps_nl,
        "delta_pct": 100.0 * (tps_nl - tps_all) / tps_all,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="assets/traj_4gpu_canonical.jsonl.gz")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--pin", type=int, default=800)
    ap.add_argument("--caps", default="64,64,64,64")
    ap.add_argument("--resid-window", type=int, default=1)
    ap.add_argument("--eta-cap", type=float, default=0.0,
                    help="MEASURED accounting overlap (MILESTONE_C ~0.0-0.3)")
    ap.add_argument("--eta-decision", type=float, default=None,
                    help="overlap the solver BELIEVES (default = eta-cap = honest)")
    ap.add_argument("--cpu-node", type=int, default=3)
    # ONLY w_numa is trained. resid_tax + w_resid + w_hotness are HEURISTIC constants
    # (held fixed at their engine defaults; a weight on a prediction can't be trained).
    ap.add_argument("--resid-tax", type=float, default=960.0,
                    help="HEURISTIC constant churn tax (µs), fixed (engine default)")
    ap.add_argument("--w-resid", type=float, default=1.0,
                    help="HEURISTIC recur=1 protect-term weight, fixed")
    ap.add_argument("--w-hotness", type=float, default=0.0,
                    help="HEURISTIC freq/EMA weight, fixed")
    ap.add_argument("--grid-numa", default="0,200,600,1500",
                    help="the ONLY trained param: w_numa grid (static home-node fact)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge-calib", default=None,
                    help="base LoaderConstants calibration JSON; write it with the "
                         "trained place_sum_weights block injected to --merge-out")
    ap.add_argument("--merge-out", default=None,
                    help="output calibration JSON (base + trained place_sum_weights)")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",")]
    eta_dec = args.eta_decision if args.eta_decision is not None else args.eta_cap
    rows = load_traj(args.trace)
    ntok = len(rows) // sim.NLAYERS
    by_layer = [[[] for _ in range(ntok)] for _ in range(sim.NLAYERS)]
    for i, r in enumerate(rows):
        t, L = i // sim.NLAYERS, i % sim.NLAYERS
        if t < ntok:
            by_layer[L][t] = [e["expert_idx"] for e in r["experts"]]
    freq = sim.build_freq(by_layer, ntok)
    fhot = max(freq.values()) if freq else 1.0
    pinned = sim.build_pinned(freq, args.pin)
    home_of = lambda e: e % 4  # NUMA home = e % tp (engine expert_home_node proxy)

    print(f"trace {args.trace}: {ntok} tokens, {sim.NLAYERS} layers, B={args.B}")
    print(f"regime: pin={args.pin} caps={caps} resid_window={args.resid_window} "
          f"eta_account={args.eta_cap} eta_decision={eta_dec} fhot={fhot}")
    print(f"HEURISTIC (fixed): resid_tax={args.resid_tax} w_resid={args.w_resid} "
          f"w_hotness={args.w_hotness} | TRAINED: w_numa (grid {args.grid_numa})")

    # champion baseline (allgpu, honest) for the delta reference
    base = sim.run_B_honest(by_layer, ntok, args.B, "allgpu", None, freq, M=4,
                            resid_window=args.resid_window, warm=0, pinned=pinned)
    # override CAPS used inside sim.run_B_honest via module global
    sim.CAPS = caps
    base = sim.run_B_honest(by_layer, ntok, args.B, "allgpu", None, freq, M=4,
                            resid_window=args.resid_window, warm=0, pinned=pinned)
    print(f"champion (allgpu honest): tps={base['tps_all']:.3f} hit={base['hit']:.4f}")

    gn = [float(x) for x in args.grid_numa.split(",")]
    results = []
    for wn in gn:  # the ONLY trained dimension
        r = run_B_priced_engine(
            by_layer, ntok, args.B, args.resid_tax, args.w_resid, wn, args.w_hotness,
            freq, fhot, home_of, args.cpu_node, eta_dec, args.eta_cap, M=4,
            resid_window=args.resid_window, pinned=pinned, caps=caps)
        d = 100.0 * (r["tps_nl"] - base["tps_all"]) / base["tps_all"]
        results.append({"w_numa": wn, "delta_vs_champ": d, **r})
    results.sort(key=lambda r: -r["delta_vs_champ"])
    print(f"\n{'rank':>4} {'w_numa':>7} {'off':>6} {'%recur':>7} {'hit':>7} {'dVSchamp':>9}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>4} {r['w_numa']:>7g} {r['off_total']:>6d} "
              f"{100*r['recur_off_frac']:>6.1f}% {r['hit']:>7.4f} "
              f"{r['delta_vs_champ']:>+8.3f}%")
    best = results[0]
    print(f"\nBEST (trained): w_numa={best['w_numa']} (heuristics resid_tax="
          f"{args.resid_tax} w_resid={args.w_resid} w_hotness={args.w_hotness}) → "
          f"off={best['off_total']} hit={best['hit']:.4f} "
          f"Δ={best['delta_vs_champ']:+.3f}% vs champion")
    if args.out:
        art = {
            "_comment": "Trained never-lose place_cons weight — ONLY w_numa (a static "
                        "home-node fact); residency+hotness are HEURISTICS (a weight on "
                        "a mis-predicting signal can't be trained). residency-HONEST "
                        "closed-loop sim fit (INV-LOADER-OBJECTIVE-MYOPIC). Loaded into "
                        "LoaderConstants.place_sum_weights; engine reads it via "
                        "loader_place_sum.",
            "w_numa": best["w_numa"],
            "heuristics_fixed": {"resid_tax": args.resid_tax, "w_resid": args.w_resid,
                                 "w_hotness": args.w_hotness},
            "provenance": {
                "generated_by": "tools .../train_place_sum_weights.py",
                "objective": "residency-honest tps_nl (cache-exclusion ground truth)",
                "trace": args.trace, "B": args.B, "pin": args.pin, "caps": caps,
                "resid_window": args.resid_window, "eta_account": args.eta_cap,
                "eta_decision": eta_dec,
                "sim_delta_vs_champion_pct": round(best["delta_vs_champ"], 4),
                "sim_offload_count": best["off_total"],
                "sim_recur_off_frac": round(best["recur_off_frac"], 4),
                "sim_hit": round(best["hit"], 4),
            },
        }
        with open(args.out, "w") as fh:
            json.dump(art, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {args.out}")

    if args.merge_calib and args.merge_out:
        calib = json.load(open(args.merge_calib))
        calib["place_sum_weights"] = {"w_numa": best["w_numa"]}  # ONLY w_numa trained
        with open(args.merge_out, "w") as fh:
            json.dump(calib, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.merge_out} (base calib + trained place_sum_weights: "
              f"w_numa={best['w_numa']})")


if __name__ == "__main__":
    main()
