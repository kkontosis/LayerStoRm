#!/usr/bin/env python3
"""Clean per-GPU LRU simulator for the ACT hit-rate decomposition (Q3/Q4).

A clean LRU cache per GPU, 460 slots, key=(layer,expert_idx).
A lookup at GPU g is a HIT iff (layer,expert_idx) is resident in g's LRU at
lookup time. After the lookup the entry is touched (MRU) / inserted (evicting
the LRU victim when full). This is the canonical "what a textbook 460-slot LRU
would have retained" model — independent of the engine's actual eviction.
"""
import csv
from collections import OrderedDict

CAP = 460

def load(f):
    with open(f) as fh:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(fh)]

def sim(rows, place):
    """place(row) -> gpu index. Returns (hit_rate, per_gpu_lookups)."""
    cache = {0: OrderedDict(), 1: OrderedDict()}
    hits = 0
    for r in rows:
        g = place(r)
        key = (r['layer'], r['expert_idx'])
        c = cache[g]
        if key in c:
            hits += 1
            c.move_to_end(key)
        else:
            c[key] = True
            if len(c) > CAP:
                c.popitem(last=False)
    return hits / len(rows)

base = load('assets/oracle_baseline.csv')
act = load('assets/oracle_act.csv')

A = sim(base, lambda r: r['oj'])            # baseline routing + e%tp placement
B = sim(act,  lambda r: r['oj'])            # ACT routing + e%tp placement
C = sim(act,  lambda r: r['j'])             # ACT routing + ACT placement

D = sum((r['cached0'] if r['j'] == 0 else r['cached1']) for r in act) / len(act)
A_eng = sum((r['cached0'] if r['j'] == 0 else r['cached1']) for r in base) / len(base)

print("=== Clean-LRU decomposition (CAP=%d/gpu) ===" % CAP)
print("A  baseline routing + e%%tp place + clean LRU : %.4f  (engine baseline=%.4f)" % (A, A_eng))
print("B  ACT routing      + e%%tp place + clean LRU : %.4f" % B)
print("C  ACT routing      + ACT place  + clean LRU : %.4f" % C)
print("D  engine actual (mean cached@j, oracle_act) : %.4f" % D)
print()
print("routing  drift  (B-A) : %+.4f" % (B - A))
print("placement effect(C-B) : %+.4f" % (C - B))
print("eviction effect (D-C) : %+.4f" % (D - C))
print("total           (D-A) : %+.4f" % (D - A))

# Q4: the 0.73 "bug" reproduction — using oracle j AND seeding the cache from the
# engine residency, or check-without-insert, etc. Try the naive "circular" variants.
# Variant X: report engine cached@j directly is D. Variant: clean LRU but lookup at j
# while ALSO inserting at oj home (split) -> not it. Try: hit iff resident at EITHER
# gpu (cross-gpu duplicate counting).
def sim_either(rows, place):
    cache = {0: OrderedDict(), 1: OrderedDict()}
    hits = 0
    for r in rows:
        g = place(r)
        key = (r['layer'], r['expert_idx'])
        resident = key in cache[0] or key in cache[1]
        if resident:
            hits += 1
        c = cache[g]
        if key in c:
            c.move_to_end(key)
        else:
            c[key] = True
            if len(c) > CAP:
                c.popitem(last=False)
    return hits / len(rows)
print()
print("Q4 variant (hit iff resident on EITHER gpu, ACT j place): %.4f" % sim_either(act, lambda r: r['j']))
