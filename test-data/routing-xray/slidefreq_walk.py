#!/usr/bin/env python3
# Sliding-window-frequency residency + resident-only self-speculation.
#
# Cache model: the resident set S is the top 20*L (layer,expert) pairs by
# usage count over the trailing H COMMITTED tokens (true top-8 counts),
# recomputed after each verify/commit. Fetch accounting:
#   - draft (policy i, resident-only): 0 fetches by construction
#   - verify: true-top-8 union of the window minus physically-present experts
#   - retention churn: promoting an expert into S that is not physically
#     present (resident or just-fetched) costs a fetch
# Acceptance walk identical to accept_walk.py: accept the longest prefix of
# the W-token window whose routing-fidelity score clears the criterion,
# commit k+1 (verify bonus), advance. Criteria are routing-level proxies
# (upper bounds on real token acceptance).
#
# Usage: slidefreq_walk.py <topk.npz> [H]
import sys
from collections import deque

import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else
            "route_dump_10k_k32_2026-07-10.topk.npz")
ids, wn = d["ids"], d["w"]
T, L, K = ids.shape
CAP = 20 * L
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
WARMUP = 1024
BASELINE = 6.14  # B=1 LRU reference from simulate_self_spec.py

def key(l, e):
    return (l << 9) | int(e)

class SlideFreq:
    def __init__(self):
        self.counts = np.zeros(L * 512, dtype=np.int32)
        self.hist = deque()
        self.S = set()
    def commit_token(self, t):
        for l in range(L):
            for e in ids[t, l]:
                self.counts[key(l, e)] += 1
        self.hist.append(t)
        while len(self.hist) > H:
            old = self.hist.popleft()
            for l in range(L):
                for e in ids[old, l]:
                    self.counts[key(l, e)] -= 1
    def retarget(self, present_extra):
        # new resident set = top-CAP by trailing count; promotions of experts
        # not physically present cost fetches
        nz = np.nonzero(self.counts)[0]
        if len(nz) > CAP:
            top = nz[np.argpartition(-self.counts[nz], CAP)[:CAP]]
        else:
            top = nz
        newS = set(top.tolist())
        promo = sum(1 for kk in newS
                    if kk not in self.S and kk not in present_extra)
        self.S = newS
        return promo

def walk(W, crit, policy="i8"):
    cache = SlideFreq()
    committed = windows = 0
    draft_f = verify_f = promo_f = 0
    acc_ks, mass_all, hits = [], [], []
    p = 0
    while p + W <= T:
        counted = p >= WARMUP
        S = cache.S
        scores = []
        for t in range(p, p + W):
            top1_ok = True
            mass_sum = 0.0
            for l in range(L):
                wrow = wn[t, l]
                if policy == "i8":
                    sel = [j for j in range(K) if key(l, ids[t, l, j]) in S]
                    fetch = 0
                else:  # ii8: top-1 fetched if absent
                    sel = [j for j in range(K) if key(l, ids[t, l, j]) in S]
                    fetch = 0
                    if 0 not in sel:
                        sel = sorted(set(sel) | {0})
                        fetch = 1
                if counted:
                    draft_f += fetch
                if 0 not in sel:
                    top1_ok = False
                mass_sum += float(wrow[sel].sum()) if sel else 0.0
            mean_mass = mass_sum / L
            if counted:
                mass_all.append(mean_mass)
            scores.append(1.0 if (crit == "top1all" and top1_ok) else
                          (mean_mass if crit != "top1all" else 0.0))
        # verify (paid for the whole window)
        fetched_now = set()
        for l in range(L):
            for t in range(p, p + W):
                for e in ids[t, l]:
                    kk = key(l, e)
                    if kk not in S and kk not in fetched_now:
                        fetched_now.add(kk)
                        if counted:
                            verify_f += 1
        # accepted prefix
        thr = 1.0 if crit == "top1all" else crit
        k = 0
        while k < W and scores[k] >= thr:
            k += 1
        for t in range(p, p + k + 1):
            if t < T:
                cache.commit_token(t)
        promo = cache.retarget(fetched_now)
        if counted:
            promo_f += promo
            hits.extend(1 if s >= thr else 0 for s in scores)
            acc_ks.append(k)
            committed += min(k + 1, T - p)
            windows += 1
        p += k + 1
    fpc = (draft_f + verify_f + promo_f) / (committed * L)
    return dict(k=float(np.mean(acc_ks)), committed=committed / windows,
                fpc=fpc, hit=100 * float(np.mean(hits)),
                mass_p50=float(np.median(mass_all)),
                mass_p90=float(np.percentile(mass_all, 90)),
                promo=promo_f / (committed * L))

# baseline: no speculation, slide-freq cache, decode B=1 and batch-W
def baseline(W):
    cache = SlideFreq()
    fetches = tokens = 0
    p = 0
    while p + W <= T:
        counted = p >= WARMUP
        fetched_now = set()
        for l in range(L):
            for t in range(p, p + W):
                for e in ids[t, l]:
                    kk = key(l, e)
                    if kk not in cache.S and kk not in fetched_now:
                        fetched_now.add(kk)
                        if counted:
                            fetches += 1
        for t in range(p, p + W):
            cache.commit_token(t)
        promo = cache.retarget(fetched_now)
        if counted:
            fetches += promo
            tokens += W
        p += W
    return fetches / (tokens * L)

print(f"cache = top-{CAP} (layer,expert) by trailing-{H}-token frequency "
      f"(~{CAP*27.6/1024:.0f} GB); LRU B=1 reference {BASELINE}")
for W in [1, 8, 32]:
    print(f"baseline no-spec batch W={W:>2}: {baseline(W):.2f} fetches/token/layer")

CRITS = [("top1all", "top1@all-L (max optimistic)"),
         (0.30, "mean-mass>=0.30"), (0.50, "mean-mass>=0.50"),
         (0.70, "mean-mass>=0.70")]
print(f"\n{'policy':>18} {'W':>3} {'criterion':>28} {'mean k':>7} {'committed':>9} "
      f"{'f/committed':>11} {'hit%':>6} {'massP50':>8} {'massP90':>8} {'promo':>6}")
for policy in ["i8", "ii8"]:
    for W in [8, 32]:
        for crit, clabel in CRITS:
            r = walk(W, crit, policy)
            flag = "  <-- BEATS B=1" if r["fpc"] < BASELINE else ""
            print(f"{policy:>18} {W:>3} {clabel:>28} {r['k']:>7.2f} {r['committed']:>9.2f} "
                  f"{r['fpc']:>11.2f} {r['hit']:>6.1f} {r['mass_p50']:>8.2f} "
                  f"{r['mass_p90']:>8.2f} {r['promo']:>6.2f}{flag}", flush=True)
