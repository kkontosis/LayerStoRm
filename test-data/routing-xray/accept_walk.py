#!/usr/bin/env python3
# Measured-acceptance walk for the self-speculation scheme, over the routing
# dump. For each draft token we compute a routing-fidelity score from the true
# gate weights (dump), accept the longest prefix of the window whose score
# clears a criterion, commit k+1 (verify bonus), and advance — so accepted
# lengths come from the ACTUAL token stream (fidelity autocorrelation and the
# evolving LRU cache included), not an iid model.
#
# Criteria (routing-level proxies; real token acceptance is <= these):
#   top1all      : draft selection includes the true top-1 expert at ALL layers
#   mass>=θ      : mean-over-layers covered true gate mass >= θ
#
# Usage: accept_walk.py <topk.npz>
import sys
from collections import OrderedDict

import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else
            "route_dump_10k_k32_2026-07-10.topk.npz")
ids, wn = d["ids"], d["w"]
T, L, K = ids.shape
CAP = 20 * L
BASELINE = 6.14  # measured B=1 LRU fetches/token/layer (simulate_self_spec)
WARMUP = 1024

class Lru:
    __slots__ = ("od", "cap")
    def __init__(self, cap):
        self.od = OrderedDict()
        self.cap = cap
    def has(self, k):
        return k in self.od
    def touch(self, k):
        self.od.move_to_end(k)
    def insert(self, k):
        if k in self.od:
            self.od.move_to_end(k)
            return
        if len(self.od) >= self.cap:
            self.od.popitem(last=False)
        self.od[k] = None

def key(l, e):
    return (l << 9) | int(e)

def draft_select(policy, lru, l, t, wrow):
    res = [j for j in range(K) if lru.has(key(l, ids[t, l, j]))]
    if policy == "i8":
        return res, []
    if policy == "ii8":
        if 0 in res:
            return res, []
        return sorted(set(res) | {0}), [key(l, ids[t, l, 0])]
    tau_res, tau_fetch = policy
    sel, fetch = [], []
    for j in range(K):
        if len(sel) == 4:
            break
        kk = key(l, ids[t, l, j])
        if lru.has(kk):
            if wrow[j] >= tau_res:
                sel.append(j)
        elif wrow[j] >= tau_fetch:
            sel.append(j)
            fetch.append(kk)
    if not sel:
        kk = key(l, ids[t, l, 0])
        sel = [0]
        if not lru.has(kk):
            fetch = [kk]
    return sel, fetch

def walk(policy, W, crit):
    lru = Lru(CAP)
    committed = windows = 0
    draft_f = verify_f = 0
    acc_ks = []
    mass_all = []
    hits = []
    p = 0
    while p + W <= T:
        counted = p >= WARMUP
        # draft W tokens from position p; per-token fidelity score
        scores = []
        for t in range(p, p + W):
            top1_ok = True
            mass_sum = 0.0
            for l in range(L):
                wrow = wn[t, l]
                sel, fetch = draft_select(policy, lru, l, t, wrow)
                for j in sel:
                    kk = key(l, ids[t, l, j])
                    if lru.has(kk):
                        lru.touch(kk)
                for kk in fetch:
                    lru.insert(kk)
                if counted:
                    draft_f += len(fetch)
                if 0 not in sel:
                    top1_ok = False
                mass_sum += float(wrow[sel].sum()) if sel else 0.0
            mean_mass = mass_sum / L
            if counted:
                mass_all.append(mean_mass)
            if crit == "top1all":
                scores.append(1.0 if top1_ok else 0.0)
            else:
                scores.append(mean_mass if top1_ok or not crit[1] else 0.0)
        # verify the whole window (paid regardless)
        for l in range(L):
            need = {key(l, e) for t in range(p, p + W) for e in ids[t, l]}
            for kk in need:
                if lru.has(kk):
                    lru.touch(kk)
                else:
                    lru.insert(kk)
                    if counted:
                        verify_f += 1
        # accepted prefix
        if crit == "top1all":
            k = 0
            while k < W and scores[k] >= 1.0:
                k += 1
        else:
            theta = crit[0]
            k = 0
            while k < W and scores[k] >= theta:
                k += 1
        if counted:
            thr = 1.0 if crit == "top1all" else crit[0]
            hits.extend(1 if s >= thr else 0 for s in scores)
            acc_ks.append(k)
            committed += k + 1
            windows += 1
        p += k + 1
    fpc = (draft_f + verify_f) / (committed * L)
    return dict(k=float(np.mean(acc_ks)), committed=committed / windows,
                fpc=fpc, hit=100 * float(np.mean(hits)),
                mass_p50=float(np.median(mass_all)), mass_p90=float(np.percentile(mass_all, 90)))

CRITS = [("top1all", "top1@all-L (max optimistic)"),
         ((0.30, False), "mean-mass>=0.30"),
         ((0.50, False), "mean-mass>=0.50"),
         ((0.70, False), "mean-mass>=0.70")]
POLICIES = [("i8", "i: resident top-8"), ("ii8", "ii: top1-fetch + res"),
            ((0.15, 0.40), "iii: 0.15/0.40")]

print(f"break-even committed/window vs B=1({BASELINE}): W=8: {8*5.82/BASELINE:.1f}? "
      f"— shown per row as fetches/committed (target < {BASELINE})")
print(f"{'policy':>22} {'W':>3} {'criterion':>28} {'mean k':>7} {'committed':>9} "
      f"{'f/committed':>11} {'massP50':>8} {'massP90':>8}")
for pol, plabel in POLICIES:
    for W in [8, 32]:
        for crit, clabel in CRITS:
            r = walk(pol, W, crit)
            flag = " <-- BEATS B=1" if r["fpc"] < BASELINE else ""
            print(f"{plabel:>22} {W:>3} {clabel:>28} {r['k']:>7.2f} {r['committed']:>9.2f} "
                  f"{r['fpc']:>11.2f} {r['mass_p50']:>8.2f} {r['mass_p90']:>8.2f}{flag}")
