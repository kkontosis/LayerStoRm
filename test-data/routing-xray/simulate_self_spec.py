#!/usr/bin/env python3
# Self-speculation fetch-economics simulator over an LS_DRIFT_DUMP capture.
#
# Scheme: decode W tokens with a DEGRADED MoE that prefers cache-resident
# experts (draft phase, ~0-1 fetches/token/layer), then verify all W in one
# batch with true top-8 routing (fetch cost ~ U(W) minus what draft already
# touched). Cache: 20*L (layer,expert) slots, ONE global LRU pool.
#
# Simplifications (stated):
#  - Policies select among each token's TRUE top-8 only (a resident 9th-best
#    substitute is not modeled; it would carry ~0 true gate mass anyway).
#  - All W draft tokens are assumed accepted; scale per-token costs by 1/alpha
#    for an assumed acceptance rate.
#  - Token-level acceptance is proxied by routing fidelity: top-1-covered %
#    and true-gate-mass covered (hidden-state divergence is not simulatable
#    from a routing dump).
#
# Usage:
#   simulate_self_spec.py <dump.bin>          # extracts topk.npz next to dump, runs sim
#   simulate_self_spec.py <topk.npz>          # runs sim directly
import os, struct, sys
from collections import OrderedDict, defaultdict

import numpy as np

CACHE_PER_LAYER = 20
WINDOWS = [8, 16, 32]
WARMUP_TOKENS = 1024
TAUS_III = [(0.10, 0.30), (0.15, 0.40)]  # (tau_res, tau_fetch), clamp 1..4

# ── extraction: dump → ids[T,L,8] int16, w[T,L,8] f32 (per-token normalized) ──
def extract(dump_path, npz_path):
    SC = 2048
    first_gpu = None
    tok_off = defaultdict(int)
    parts = defaultdict(list)
    with open(dump_path, "rb") as f:
        while True:
            hdr = f.read(24)
            if len(hdr) < 24:
                break
            _, layer, gpu, nt, ne, tk = struct.unpack("<6i", hdr)
            f.seek(nt * ne * 4, 1)
            idx = np.frombuffer(f.read(nt * tk * 4), dtype="<i4").reshape(nt, tk)
            w = np.frombuffer(f.read(nt * tk * 4), dtype="<f4").reshape(nt, tk)
            if first_gpu is None:
                first_gpu = gpu
            if gpu != first_gpu:
                continue
            parts[layer].append((idx, w))
            tok_off[layer] += nt
    layers = sorted(parts)
    T = min(tok_off[l] for l in layers)
    L, K = len(layers), 8
    ids = np.zeros((T, L, K), dtype=np.int16)
    wn = np.zeros((T, L, K), dtype=np.float32)
    for li, l in enumerate(layers):
        i = np.concatenate([p[0] for p in parts[l]], axis=0)[:T]
        v = np.concatenate([p[1] for p in parts[l]], axis=0)[:T].astype(np.float32)
        s = v.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        ids[:, li] = i
        wn[:, li] = v / s
    np.savez_compressed(npz_path, ids=ids, w=wn)
    print(f"extracted T={T} L={L} -> {npz_path}")
    return ids, wn

path = sys.argv[1] if len(sys.argv) > 1 else "route_dump_10k_k32_2026-07-10.bin"
if path.endswith(".npz"):
    d = np.load(path)
    ids, wn = d["ids"], d["w"]
else:
    npz = os.path.splitext(path)[0] + ".topk.npz"
    if os.path.exists(npz):
        d = np.load(npz)
        ids, wn = d["ids"], d["w"]
    else:
        ids, wn = extract(path, npz)
T, L, K = ids.shape
CAP = CACHE_PER_LAYER * L
print(f"T={T} tokens, L={L} layers, cache={CAP} global-LRU slots "
      f"(= {CACHE_PER_LAYER}*L avg, ~{CAP*27.6/1024:.0f} GB at 27.6 MB/expert)")

class Lru:
    __slots__ = ("od", "cap")
    def __init__(self, cap):
        self.od = OrderedDict()
        self.cap = cap
    def has(self, k):
        return k in self.od
    def touch(self, k):  # use (must be resident)
        self.od.move_to_end(k)
    def insert(self, k):  # fetch
        if k in self.od:
            self.od.move_to_end(k)
            return
        if len(self.od) >= self.cap:
            self.od.popitem(last=False)
        self.od[k] = None

def key(l, e):
    return (l << 9) | int(e)

# draft policy: returns (selected_slots list of col indices 0..7, fetched keys)
def draft_select(policy, lru, l, tid, wrow):
    res = [j for j in range(K) if lru.has(key(l, ids[tid, l, j]))]
    if policy == "i8":
        return res, []
    if policy == "i4":
        return res[:4] if len(res) > 4 else res, []   # rows are weight-ordered
    if policy == "ii8":
        if 0 in res:
            return res, []
        return sorted(set(res) | {0}), [key(l, ids[tid, l, 0])]
    # iii: dual bar, clamp 1..4
    tau_res, tau_fetch = policy
    sel, fetch = [], []
    for j in range(K):
        if len(sel) == 4:
            break
        kk = key(l, ids[tid, l, j])
        if lru.has(kk):
            if wrow[j] >= tau_res:
                sel.append(j)
        elif wrow[j] >= tau_fetch:
            sel.append(j)
            fetch.append(kk)
    if not sel:  # at least 1: take true top-1
        kk = key(l, ids[tid, l, 0])
        sel = [0]
        if not lru.has(kk):
            fetch = [kk]
    return sel, fetch

def run(policy, W, with_draft=True):
    lru = Lru(CAP)
    draft_f = verify_f = 0
    top1_cov = mass_cov = 0.0
    zero_sel = 0
    ntok = 0
    for w0 in range(0, T - W + 1, W):
        toks = range(w0, w0 + W)
        counted = w0 >= WARMUP_TOKENS
        if with_draft:
            for t in toks:
                for l in range(L):
                    wrow = wn[t, l]
                    sel, fetch = draft_select(policy, lru, l, t, wrow)
                    # touch residents BEFORE inserting fetches: an insert may
                    # evict a just-selected resident from the global pool
                    for j in sel:
                        kk = key(l, ids[t, l, j])
                        if lru.has(kk):
                            lru.touch(kk)
                    for kk in fetch:
                        lru.insert(kk)
                    if counted:
                        draft_f += len(fetch)
                        top1_cov += 1.0 if 0 in sel else 0.0
                        mass_cov += float(wrow[sel].sum()) if sel else 0.0
                        zero_sel += 0 if sel else 1
                if counted:
                    ntok += 1
        elif counted:
            ntok += W
        # verify: true top-8 for the whole window, batched
        for l in range(L):
            need = {key(l, e) for t in toks for e in ids[t, l]}
            miss = [kk for kk in need if not lru.has(kk)]
            for kk in need:
                if kk in lru.od:
                    lru.touch(kk)
            for kk in miss:
                lru.insert(kk)
            if counted:
                verify_f += len(miss)
    ntl = ntok * L
    return dict(
        draft=draft_f / ntl if with_draft else 0.0,
        verify=verify_f / ntl,
        total=(draft_f + verify_f) / ntl,
        top1=100 * top1_cov / ntl if with_draft else float("nan"),
        mass=100 * mass_cov / ntl if with_draft else float("nan"),
        zero=100 * zero_sel / ntl if with_draft else float("nan"),
    )

print(f"\n(all numbers per token per layer, tokens {WARMUP_TOKENS}.. after LRU warm-up; "
      f"x{L} for per-token totals; cold reference 8.0)")
hdr = f"{'policy':>22} {'W':>3} {'draft f':>8} {'verify f':>9} {'TOTAL f':>8} {'top1%':>6} {'mass%':>6} {'zero%':>6}"
print(hdr)

# baselines: no draft
r = run(None, 1, with_draft=False)
print(f"{'baseline B=1 LRU':>22} {1:>3} {r['draft']:>8.2f} {r['verify']:>9.2f} {r['total']:>8.2f}")
for W in WINDOWS:
    r = run(None, W, with_draft=False)
    print(f"{'baseline batch-only':>22} {W:>3} {r['draft']:>8.2f} {r['verify']:>9.2f} {r['total']:>8.2f}")

for pol, label in [("i8", "i: resident top-8"), ("i4", "i: resident top-4"),
                   ("ii8", "ii: top1-fetch + res")] + \
                  [(t, f"iii: res>={t[0]}/fetch>={t[1]}") for t in TAUS_III]:
    for W in WINDOWS:
        r = run(pol, W)
        print(f"{label:>22} {W:>3} {r['draft']:>8.2f} {r['verify']:>9.2f} {r['total']:>8.2f} "
              f"{r['top1']:>6.1f} {r['mass']:>6.1f} {r['zero']:>6.1f}")
