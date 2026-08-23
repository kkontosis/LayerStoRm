#!/usr/bin/env python3
"""TIMING-AWARE offload gate on the fixed champion trace (no fork, no GPU boot).

Follow-up to rare_node1_sim.py. The all-rare exclusion (hit 0.7114) is an idealized
ceiling that offloads >5 of 8 routed experts per layer-token — it fully SPILLS the
host FFN and cannot hide. Here the offload DECISION is TIMING-BASED: an expert is
offloaded to CPU only when its host FFN compute is HIDDEN under the GPU fetch wall
(it does NOT raise that verify-chunk-layer's wall time). This yields FEWER,
timing-justified picks and the REALISTIC hit achievable WITHOUT an ultra-slow CPU.

Faithful engine frame = the dsp52 SPECULATIVE verify chunk: per (chunk,layer) the
loader fetches the UNION of the chunk's distinct routed experts (one fetch command),
so the loader hit is over DISTINCT union members per chunk-layer (not per token). The
champion 0.3677 anchor is reproduced in THIS union frame by calibrating caps.

COST MODEL (measured, coordinator-specified):
  host_FFN   = 120 µs / offloaded expert   (fast-Q8 kernel path; NOT the 1.16 ms
               lossless — offloads run the fast dequant-GEMM, per-expert, B-folded)
  fold       = 440 µs / engaged layer      (D2H in + H2D out + barriers, fixed)
  fetch      = 960 µs / expert H2D, 4 parallel channels ⇒ per_gpu_fetch(m)=⌈m/4⌉·960
  η(B)       = min(1, 1−0.85·e^-(B-1)/4)   (early-kick overlap; ~1.0 at B≥12, 0.15 @B1)
GATE (never-lose, hidden-only): only RARE MISSES are candidates (rare⇔M3 freq<0.1·fhot;
  a HIT is already resident so offloading it saves no fetch; a non-rare miss is
  residency-load-bearing). Greedily accept k rare misses while the offloaded layer
  wall  max(per_gpu_fetch(m−k), fold+k·host − η·slack)  does NOT exceed the all-GPU
  wall per_gpu_fetch(m). k*=0 ⇒ no beneficial offload (champion-neutral).
"""
import argparse
import math
import statistics
from collections import Counter

import rare_node1_sim as S

FETCH_US = 960.0
CHANNELS = 4
FOLD_US = 440.0
HOST_US = 120.0


def eta(B):
    return min(1.0, 1.0 - 0.85 * math.exp(-(B - 1) / 4.0))


def pgf(m):
    return math.ceil(max(m, 0) / CHANNELS) * FETCH_US


def gate_k(m, cand_n, B, host_us=HOST_US, fold_us=FOLD_US):
    """Largest k in [0,cand_n] rare-miss offloads whose wall ≤ all-GPU wall (never
    raises the layer wall ⇒ host FFN hidden under the fetch window)."""
    base = pgf(m)
    e = eta(B)
    best = 0
    for k in range(1, cand_n + 1):
        gpu2 = pgf(m - k)
        cpu2 = fold_us + k * host_us
        exposed = max(0.0, cpu2 - e * gpu2)
        wall = gpu2 + exposed
        if wall <= base + 1e-9:
            best = k
    return best


def build_pinned(freq, npin):
    """M3-static hot set: top-npin (layer,expert) by freq, always HBM-resident
    (LS_ARENA_PLACE_FREQ) ⇒ always a hit, never rare, never offloaded."""
    if npin <= 0:
        return set()
    return {k for k, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:npin]}


def run_union(by_layer, layers, ntok, caps, B, freq, thr, gate=True,
              host_us=HOST_US, fold_us=FOLD_US, node_frac=None, pinned=None):
    if pinned is None:
        pinned = set()
    caches = [S.Lru(caps[g]) for g in range(S.M)]
    lookups = hits = 0
    off_counts = []          # per engaged (chunk,layer): k offloaded
    engaged = 0
    off_total = 0
    tot_wall_all = tot_wall_off = 0.0
    excl_uses = excl_hits_a = 0   # A-side accounting of the offloaded uses
    nchunks = math.ceil(ntok / B)
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        bsz = t1 - t0
        for L in layers:
            uni = {}
            for t in range(t0, t1):
                for (ex, j) in by_layer[L][t]:
                    if ex not in uni:
                        uni[ex] = j
            resident = []
            miss = []
            for ex, j in uni.items():
                g = j if 0 <= j < S.M else ex % S.M
                if (L, ex) in pinned or (L, ex) in caches[g].od:
                    resident.append(ex)
                else:
                    miss.append(ex)
            m = len(miss)
            K = set()
            if gate:
                cand = [ex for ex in miss
                        if freq.get((L, ex), 0) < thr
                        and (node_frac is None or S.det_node1(ex, node_frac))]
                k = gate_k(m, len(cand), bsz, host_us, fold_us)
                K = set(cand[:k])
            k = len(K)
            if k:
                engaged += 1
                off_counts.append(k)
                off_total += k
            # wall accounting
            base = pgf(m)
            gpu2 = pgf(m - k)
            cpu2 = fold_us + k * host_us if k else 0.0
            exposed = max(0.0, cpu2 - eta(bsz) * gpu2) if k else 0.0
            tot_wall_all += base
            tot_wall_off += (gpu2 + exposed) if k else base
            # residency advance over the union (offloaded excluded from GPU)
            for ex, j in uni.items():
                g = j if 0 <= j < S.M else ex % S.M
                key = (L, ex)
                if key in pinned:
                    lookups += 1
                    hits += 1
                    continue
                if ex in K:
                    excl_uses += 1  # it was a miss ⇒ 0 A-side hits by construction
                    caches[g].evict(key)
                    continue
                lookups += 1
                if key in caches[g].od:
                    hits += 1
                    caches[g].od.move_to_end(key)
                else:
                    caches[g].insert(key)
    return {
        "hit": hits / max(lookups, 1), "lookups": lookups, "misses": lookups - hits,
        "off_total": off_total, "off_per_100tok": off_total,
        "engaged": engaged, "off_counts": off_counts,
        "wall_all": tot_wall_all, "wall_off": tot_wall_off,
        "wall_delta_pct": 100.0 * (tot_wall_all - tot_wall_off) / tot_wall_all,
        "excl_uses": excl_uses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16, help="verify-chunk width (1+gamma)")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--rarity", type=float, default=0.1)
    ap.add_argument("--host-us", type=float, default=HOST_US)
    ap.add_argument("--node-frac", type=float, default=None,
                    help="restrict to node-1 spill proxy fraction (None=rarity only)")
    ap.add_argument("--pin", type=int, default=3400,
                    help="M3-static hot set pinned resident (calibrates the anchor)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--sweep-thresh", action="store_true",
                    help="retrain: sweep the rarity threshold for the timing objective")
    args = ap.parse_args()

    freq, fhot = S.load_freq(S.FREQ_TABLE)
    by, layers, ntok = S.load_trace(S.TRACE)
    thr = args.rarity * fhot
    cap = args.cap or 600
    caps = [cap] * S.M
    pinned = build_pinned(freq, args.pin)
    print(f"UNION frame B={args.B} (η={eta(args.B):.3f}), host={args.host_us}us/expert, "
          f"fold={FOLD_US}us, fhot={fhot}; cap={cap} pin={args.pin} rare⇔freq<{thr:g}")

    if args.calibrate:
        print("champion (gate OFF) union hit vs (cap,pin) — target 0.3677:")
        for pin in (3000, 3200, 3400, 3600, 4000):
            r = run_union(by, layers, ntok, caps, args.B, freq, thr, gate=False,
                          pinned=build_pinned(freq, pin))
            print(f"  cap={cap} pin={pin:5d}  hit={r['hit']:.4f}  lookups={r['lookups']}")
        return

    champ = run_union(by, layers, ntok, caps, args.B, freq, thr, gate=False, pinned=pinned)
    print(f"\nCHAMPION (union, gate OFF): hit={champ['hit']:.4f} lookups={champ['lookups']}"
          f"   [anchor 0.3677]")

    if args.sweep_thresh:
        print("\nRETRAIN — rarity threshold sweep (rare∧node1-25% timing gate):")
        print("  thr(×fhot)  offloads/100tok  hit     Δhit    wallΔ%")
        for rt in (0.05, 0.10, 0.15, 0.25, 0.40):
            tg = run_union(by, layers, ntok, caps, args.B, freq, rt * fhot, gate=True,
                           host_us=args.host_us, node_frac=0.25, pinned=pinned)
            print(f"  {rt:.2f}        {tg['off_total']:6d}          {tg['hit']:.4f}  "
                  f"{tg['hit']-champ['hit']:+.4f}  {tg['wall_delta_pct']:+.1f}")
        return

    for name, nf in (("rarity-only", None), (f"rare∧node1-25% (engine e01 set)", 0.25)):
        tg = run_union(by, layers, ntok, caps, args.B, freq, thr, gate=True,
                       host_us=args.host_us, node_frac=nf, pinned=pinned)
        oc = tg["off_counts"]
        h = Counter(oc)
        hh = " ".join(f"{k}:{h[k]}" for k in sorted(h))
        print(f"\n[timing-gate {name}]")
        print(f"  offloads / 100 tok      : {tg['off_total']}  ({tg['off_total']/ntok:.2f}/tok)")
        print(f"  engaged (chunk,layer)   : {tg['engaged']}  mean/engaged="
              f"{statistics.mean(oc) if oc else 0:.2f}  max={max(oc) if oc else 0}")
        print(f"  per-engaged-layer hist  : {hh}")
        print(f"  REALISTIC hit           : {tg['hit']:.4f}  (champion {champ['hit']:.4f}, "
              f"idealized all-rare 0.7114, engine e01 0.2035)")
        print(f"  Δhit vs champion        : {tg['hit']-champ['hit']:+.4f}")
        print(f"  modeled MoE-fetch wall Δ: {tg['wall_delta_pct']:+.2f}%")


if __name__ == "__main__":
    main()
