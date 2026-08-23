#!/usr/bin/env python3
"""TASK A/B — short-horizon residency reframe (CHEAP, no-boot).

Resolves: does pricing CPU-offload residency-safety at the SHORT cache-
RETENTION horizon (not the 16-token recurrence rate) open a surviving win,
and can a FREE signal predict the safe set?

Uses the SAME canonical trajectory + cost model as cpu_offload_bsweep.py.
The honest sim's token-by-token LRU ALREADY prices residency correctly (an
offloaded expert's re-use is an extra miss ONLY if it was still resident in
the OFF baseline). What this script adds:
  1. EFFECTIVE RETENTION WINDOW: empirical survival curve P(hit | reuse-gap)
     from the real baseline cache — the gap beyond which a re-use would have
     MISSED ANYWAY (⇒ offloading is free).
  2. RESIDENCY-SAFE POPULATION at k=2,4 (and the measured window): fraction of
     expert-uses whose next-use gap > k ⇒ safe to offload at ZERO residency cost.
  3. CORRECTED RE-SWEEP at B=16: offload the residency-safe set (ORACLE next-use,
     then the FREE prev-union signal) and report the wall delta vs the exposed-
     overlap knob eta — does it cross neutral-or-positive?
"""
import argparse
import math
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trajectory_sim import load_traj  # noqa: E402
import cpu_offload_bsweep as S  # noqa: E402

NL = S.NLAYERS
M = 4


def load_by_layer(trace):
    rows = load_traj(trace)
    ntok = len(rows) // NL
    by_layer = [[[] for _ in range(ntok)] for _ in range(NL)]
    for i, r in enumerate(rows):
        t, L = i // NL, i % NL
        if t < ntok:
            by_layer[L][t] = [e["expert_idx"] for e in r["experts"]]
    return by_layer, ntok


# ── 1. effective retention window: survival curve P(hit | reuse gap) ──────────
def retention_survival(by_layer, ntok, caps, warm=0, pinned=None):
    """CHAMPION cache (pin + per-home LRU) advanced token-by-token. For every
    reuse of (layer,expert), record (gap, hit?), SPLIT by pinned membership.
    Pinned experts are always resident; the offload CANDIDATES are the cold
    (non-pinned) misses, so their retention window is what actually matters.
    Returns per-gap hit prob for the COLD (non-pinned) population + hit rates."""
    if pinned is None:
        pinned = set()
    cachelist = [S.Lru(caps[g]) for g in range(M)]
    if warm:
        S.warm_caches(cachelist, by_layer, ntok, warm, M)
    last_use = {}
    gap_hit = defaultdict(lambda: [0, 0])   # cold-only gap -> [hits, total]
    lookups = hits = 0
    cold_lookups = cold_hits = 0
    for t in range(ntok):
        for layer in range(NL):
            for e in by_layer[layer][t]:
                key = (layer, e)
                g = e % M
                lookups += 1
                if key in pinned:
                    hits += 1
                    last_use[key] = t
                    continue
                is_hit = cachelist[g].hit(key)
                cold_lookups += 1
                if is_hit:
                    hits += 1
                    cold_hits += 1
                else:
                    cachelist[g].insert(key)
                if key in last_use:
                    gap = t - last_use[key]
                    gap_hit[gap][0] += int(is_hit)
                    gap_hit[gap][1] += 1
                last_use[key] = t
    return (gap_hit, hits / max(lookups, 1),
            cold_hits / max(cold_lookups, 1))


def eff_window(gap_hit, thresh=0.5):
    """Largest gap G such that P(hit|gap<=G) still >= thresh (the retention
    horizon: reuses at gap>window would have missed anyway)."""
    win = 0
    for gap in sorted(gap_hit):
        h, n = gap_hit[gap]
        if n < 20:
            continue
        if h / n >= thresh:
            win = gap
        else:
            break
    return win


# ── 2. residency-safe population: next-use gap distribution ───────────────────
def nextuse_gaps(by_layer, ntok):
    """For every expert-use (layer,e,t) return the gap to its NEXT use of the
    same (layer,e); big sentinel if never again."""
    occ = [dict() for _ in range(NL)]
    for L in range(NL):
        for t in range(ntok):
            for e in by_layer[L][t]:
                occ[L].setdefault(e, []).append(t)
    gaps = []
    for L in range(NL):
        for e, lst in occ[L].items():
            for i in range(len(lst)):
                nxt = lst[i + 1] - lst[i] if i + 1 < len(lst) else 10**9
                gaps.append(nxt)
    return gaps


def safe_frac(gaps, window):
    """Fraction of uses whose next-use gap > window (residency-safe: would be
    evicted before reuse ⇒ offloading costs 0 extra miss). Excludes the trace-
    END never-again cases separately (they inflate 'safe' artifactually)."""
    n = len(gaps)
    safe = sum(1 for g in gaps if g > window)
    never = sum(1 for g in gaps if g >= 10**9)
    recurs_in = sum(1 for g in gaps if g <= window)
    return dict(n=n, safe=safe, safe_pct=100 * safe / n,
                never=never, never_pct=100 * never / n,
                recurs_in_pct=100 * recurs_in / n,
                safe_ex_never=safe - never,
                safe_ex_never_pct=100 * (safe - never) / n)


# ── 3. corrected re-sweep: offload the residency-safe set, sweep eta ──────────
def resweep(by_layer, ntok, B, caps, pinned, window, eta_cap, signal,
            warm=0):
    """Offload the residency-safe misses per engaged layer/chunk and price the
    HONEST wall (fetch relief - exposed host FFN) + HONEST residency (token LRU).

    signal:
      'oracle' — offload misses with true next-use gap > window.
      'prev'   — FREE b0_prev: offload misses NOT in the previous committed
                 token's routed union for this layer (predicted non-recurring).
      'none'   — allgpu baseline.
    All offloads are wall-gated by never-lose (offload only if it lowers the
    layer wall at the given eta) so the result is a real never-lose policy."""
    S.ETA_CAP = eta_cap
    caches = [S.Lru(caps[g]) for g in range(M)]
    if warm:
        S.warm_caches(caches, by_layer, ntok, warm, M)
    # next-use oracle
    nextuse = S.build_nextuse(by_layer, ntok)
    nchunks = math.ceil(ntok / B)
    tot_all = tot_nl = 0.0
    off_total = eng = 0
    safe_candidates = 0
    gpu_lookups = gpu_hits = 0
    for c in range(nchunks):
        t0, t1 = c * B, min(c * B + B, ntok)
        bsz = t1 - t0
        te = t1 - 1
        for layer in range(NL):
            uni = {}
            for t in range(t0, t1):
                for e in by_layer[layer][t]:
                    uni[e] = 1
            U = len(uni)
            resident = {e for e in uni
                        if (layer, e) in pinned or (layer, e) in caches[e % M].od}
            miss = [e for e in uni if e not in resident]
            m = len(miss)
            base_wall = S.layer_wall_allgpu(U, m, bsz)
            # candidate safe-to-offload set among the misses
            if signal == "none":
                cand = []
            elif signal == "oracle":
                cand = [e for e in miss if nextuse(layer, e, te) > window]
            elif signal == "prev":
                prevset = set()
                if t0 - 1 >= 0:
                    prevset = set(by_layer[layer][t0 - 1])
                cand = [e for e in miss if e not in prevset]
            else:
                raise ValueError(signal)
            safe_candidates += len(cand)
            # never-lose greedy: offload safe cands while it lowers the wall
            eta = S.overlap_eta(bsz)
            best_w, best_k = base_wall, 0
            for k in range(1, len(cand) + 1):
                gpu2 = max(S.per_gpu_fetch(m - k, S.DMA_CONTENTION),
                           S.per_gpu_compute(U - k, bsz))
                cpu2 = S.FOLD_US + k * S.host_expert(bsz)
                exposed = max(0.0, cpu2 - eta * gpu2)
                w = gpu2 + exposed
                if w < best_w - 1e-9:
                    best_w, best_k = w, k
            K = set(cand[:best_k])
            k = len(K)
            wall = best_w if k else base_wall
            tot_all += base_wall
            tot_nl += wall
            off_total += k
            if k:
                eng += 1
            # honest token-level residency advance
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
    fixed = S.FIXED_TOK_US * ntok
    tps_all = ntok / ((tot_all + fixed) / 1e6)
    tps_nl = ntok / ((tot_nl + fixed) / 1e6)
    return dict(B=B, signal=signal, window=window, eta=eta_cap,
                off_total=off_total, eng_frac=eng / (nchunks * NL),
                safe_cand=safe_candidates,
                hit=gpu_hits / max(gpu_lookups, 1),
                tps_all=tps_all, tps_nl=tps_nl,
                delta=100 * (tps_nl - tps_all) / tps_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="assets/traj_4gpu_canonical.jsonl.gz")
    ap.add_argument("--pin", type=int, default=800)
    args = ap.parse_args()
    by_layer, ntok = load_by_layer(args.trace)
    freq = S.build_freq(by_layer, ntok)
    pinned = S.build_pinned(freq, args.pin)
    print(f"trace {args.trace}: {ntok} tokens x {NL} layers, top-8, M={M} GPUs, "
          f"pin={args.pin}")

    # ── 1. champion cache (pin+LRU): measure retention of the COLD (offloadable)
    #      population. 75 layers SHARE each e%4 home ⇒ ~150 insertions/tok/home,
    #      so the non-pinned tail has SUB-TOKEN retention; hits come from pinning.
    print("\n=== 1. EFFECTIVE CACHE-RETENTION WINDOW (champion pin+LRU) ===")
    print("caps/home  hit(all)  hit(cold-tail)  cold-retention(P(hit)>=0.5, tok)")
    chosen = None
    for cap in (32, 48, 64, 96, 128):
        caps = [cap] * M
        gap_hit, hr, cold_hr = retention_survival(by_layer, ntok, caps,
                                                  warm=0, pinned=pinned)
        w50 = eff_window(gap_hit, 0.5)
        mark = ""
        if chosen is None and cap == 64:
            chosen = (cap, caps, gap_hit, hr, cold_hr, w50)
            mark = "  <- champion caps=64 (measured engine hit 0.368)"
        print(f"  {cap:3d}     {hr:.3f}     {cold_hr:.3f}          {w50:3d}{mark}")
    cap, caps, gap_hit, hr, cold_hr, w50 = chosen
    print(f"\n  CHAMPION regime: caps={cap}/home pin={args.pin}, hit(all)={hr:.3f}, "
          f"hit(cold-tail)={cold_hr:.3f}")
    print(f"  COLD-tail retention window (P(hit)>=0.5) = {w50} tok  "
          f"(the offload candidates are cold misses)")
    print("  cold-tail survival detail (gap: P(hit), n):")
    for gap in sorted(gap_hit)[:8]:
        h, n = gap_hit[gap]
        print(f"    gap {gap:3d}: P(hit)={h/max(n,1):.3f}  n={n}")
    # the retention window used for the safe-population + re-sweep: the cold-tail
    # window if positive, else 1 tok (sub-token retention ⇒ ~any reuse misses).
    w_eff = max(w50, 1)

    # ── 2. residency-safe population ──
    print("\n=== 2. RESIDENCY-SAFE POPULATION (next-use gap > window) ===")
    gaps = nextuse_gaps(by_layer, ntok)
    print("window   safe%%(incl trace-end)  safe%%(excl never-again)  recurs-within%%")
    for win in sorted({1, 2, w_eff, 4, 8, 16}):
        s = safe_frac(gaps, win)
        print(f"  {win:3d}     {s['safe_pct']:6.1f}                {s['safe_ex_never_pct']:6.1f}"
              f"                  {s['recurs_in_pct']:6.1f}")
    print(f"  (trace-end never-again cases = {safe_frac(gaps,0)['never_pct']:.1f}% "
          f"of uses — excluded from 'safe excl' as a 100-tok trace artifact)")

    # ── 3. corrected re-sweep at B=16 ──
    print("\n=== 3. CORRECTED RE-SWEEP at B=16 (offload residency-safe set) ===")
    print(f"regime caps={cap}/home pin={args.pin}, window={w_eff} tok "
          f"(the measured cold-tail retention window)")
    print("eta_cap  signal   off_total  eng_frac  safe_cand  hit    tps_all  tps_nl  delta%")
    for eta in (0.0, 0.15, 0.3, 0.5, 1.0):
        for sig in ("oracle", "prev"):
            r = resweep(by_layer, ntok, 16, caps, pinned, w_eff, eta, sig)
            print(f"  {eta:.2f}    {sig:6s}   {r['off_total']:6d}     "
                  f"{r['eng_frac']:.3f}    {r['safe_cand']:6d}    {r['hit']:.3f}  "
                  f"{r['tps_all']:6.2f}  {r['tps_nl']:6.2f}  {r['delta']:+6.2f}")
    print("\n(oracle = true next-use>window; prev = FREE b0_prev prev-union non-member)")


if __name__ == "__main__":
    main()
