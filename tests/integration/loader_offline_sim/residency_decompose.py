#!/usr/bin/env python3
"""Isolate the PURE freed-slot residency gain of the RETAINED set from the
denominator-shift effect, on the fixed champion trace (no fork, no GPU boot).

The offload hit rise (champion 0.3586 → 0.5513 for rare∧node-1) could be either
(a) a DENOMINATOR SHIFT — excluding low-hitting rare experts mechanically lifts the
average — or (b) a REAL freed-LRU-slot residency gain for the experts that STAY. This
decomposes it. Validated B=1 token-major frame, cap=208 (reproduces champion 0.3677).

For an offloaded set O (excluded from the GPU cache, host-served, not a GPU lookup):
  (1) retained hit WITH offload    : hits/lookups over uses key∉O in the OFFLOAD run.
  (2) retained hit WITHOUT offload : hits/lookups over the SAME key∉O uses in the
                                     CHAMPION run (no exclusion) — same experts,
                                     champion residency.
  denominator shift = (2) − champion_overall   (retained set vs the full average)
  PURE residency gain = (1) − (2)               (freed-slot benefit on what stays)
  retained fetches saved / 100 tok = hits1 − hits2 (extra retained hits ⇒ fetches not
     issued; SEPARATE from the offloaded set's own fetch changes).
"""
import rare_node1_sim as S

CAP = [208] * 4
M = 4


def load():
    freq, fhot = S.load_freq(S.FREQ_TABLE)
    by, layers, ntok = S.load_trace(S.TRACE)
    return freq, by, layers, ntok


def offload_retained(by, layers, ntok, O):
    """Offload run: O excluded (evicted, host-served, not a lookup). Token-major."""
    caches = [S.Lru(CAP[g]) for g in range(M)]
    look = hits = 0
    for t in range(ntok):
        for L in layers:
            for (e, j) in by[L][t]:
                key = (L, e)
                g = j if 0 <= j < M else e % M
                if key in O:
                    caches[g].evict(key)
                    continue
                look += 1
                if key in caches[g].od:
                    hits += 1
                    caches[g].od.move_to_end(key)
                else:
                    caches[g].insert(key)
    return hits, look


def champ_on_retained(by, layers, ntok, O):
    """Champion (no exclusion); count hits/lookups only over uses whose key∉O."""
    caches = [S.Lru(CAP[g]) for g in range(M)]
    look = hits = 0
    for t in range(ntok):
        for L in layers:
            for (e, j) in by[L][t]:
                key = (L, e)
                g = j if 0 <= j < M else e % M
                h = key in caches[g].od
                if h:
                    caches[g].od.move_to_end(key)
                else:
                    caches[g].insert(key)
                if key not in O:
                    look += 1
                    if h:
                        hits += 1
    return hits, look


def main():
    freq, by, layers, ntok = load()
    occ = {(L, e) for L in layers for t in range(ntok) for (e, j) in by[L][t]}
    rare = {(L, e) for (L, e) in occ if freq.get((L, e), 0) < 3.7}
    node1 = {(L, e) for (L, e) in occ if S.det_node1(e, 0.25)}
    sets = {
        "rare∧node-1 (thr0.10)": rare & node1,
        "freq-0 (never-profiled)": {(L, e) for (L, e) in occ if freq.get((L, e), 0) == 0},
    }
    ch, cl = offload_retained(by, layers, ntok, set())
    ovr = ch / cl
    print(f"champion OVERALL (token-major cap208): hit={ovr:.4f} lookups={cl}  [validated 0.3677 knee]")
    for name, O in sets.items():
        excl = sum(1 for t in range(ntok) for L in layers for (e, j) in by[L][t]
                   if (L, e) in O)
        h1, R1 = offload_retained(by, layers, ntok, O)
        h2, R2 = champ_on_retained(by, layers, ntok, O)
        hit1, hit2 = h1 / R1, h2 / R2
        print(f"\n[{name}]  offloaded uses={excl}  retained lookups={R1}")
        print(f"  (1) retained hit WITH offload    = {hit1:.4f}")
        print(f"  (2) retained hit WITHOUT offload = {hit2:.4f}  (champion, same retained set)")
        print(f"  champion OVERALL                 = {ovr:.4f}")
        print(f"  denominator shift   (2)-overall  = {100*(hit2-ovr):+.1f} pp")
        print(f"  PURE residency gain (1)-(2)      = {100*(hit1-hit2):+.2f} pp   [freed-slot benefit]")
        print(f"  retained FETCHES SAVED / 100 tok = {h1-h2}")


if __name__ == "__main__":
    main()
