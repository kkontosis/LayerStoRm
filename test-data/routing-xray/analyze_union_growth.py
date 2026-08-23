#!/usr/bin/env python3
# Union-growth curve U(N): distinct experts fetched for an N-token contiguous
# window, from an LS_DRIFT_DUMP superchunk routing capture.
#
# Separates two effects the coverage number conflates:
#   - overlap within a batch  -> U(N) < 8N even under uniform routing
#     (uniform model: E * (1 - (1 - topk/E)^N))
#   - frequent experts (skew) -> measured U(N) below the uniform curve, and
#     residency: fetch count after excluding the PREVIOUS superchunk's top-K.
import struct, sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route_dump_sc.bin"
SC_TOKENS = 2048
WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
RESIDENT_KS = ([int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2
                else [50, 100, 152])

first_gpu = None
tok_off = defaultdict(int)
seqs = defaultdict(list)   # (layer, sc) -> list of (nt, idx array) in order
E = TK = None

with open(PATH, "rb") as f:
    while True:
        hdr = f.read(24)
        if len(hdr) < 24:
            break
        seq, layer, gpu, nt, ne, tk = struct.unpack("<6i", hdr)
        f.seek(nt * ne * 4, 1)
        idx = np.frombuffer(f.read(nt * tk * 4), dtype="<i4").reshape(nt, tk)
        f.seek(nt * tk * 4, 1)  # skip weights
        E, TK = ne, tk
        if first_gpu is None:
            first_gpu = gpu
        if gpu != first_gpu:
            continue
        sc = tok_off[layer] // SC_TOKENS
        tok_off[layer] += nt
        seqs[(layer, sc)].append(idx)

# token-major per (layer, sc)
mats = {k: np.concatenate(v, axis=0) for k, v in seqs.items()}
layers = sorted({l for (l, _) in mats})
scs = sorted({s for (_, s) in mats})
full = [s for s in scs if all(mats.get((l, s)) is not None and len(mats[(l, s)]) == SC_TOKENS for l in layers)]
print(f"layers={len(layers)} superchunks={scs} full={full} E={E} topk={TK}")

print("\n== U(N): distinct experts per N-token contiguous window (mean over windows x layers x full superchunks) ==")
print(f"{'N':>5} {'measured':>9} {'uniform':>8} {'8N cap':>7} {'meas/8N':>8} {'skew gain vs uniform':>20}")
for N in WINDOWS:
    vals = []
    for s in full:
        for l in layers:
            m = mats[(l, s)]
            nw = SC_TOKENS // N
            for w in range(nw):
                vals.append(len(np.unique(m[w * N:(w + 1) * N])))
    meas = float(np.mean(vals))
    uni = E * (1 - (1 - TK / E) ** N)
    cap = min(TK * N, E)
    print(f"{N:>5} {meas:>9.1f} {uni:>8.1f} {cap:>7} {meas/cap:>8.2f} {100*(uni-meas)/uni:>19.1f}%")

print("\n== Residency-adjusted fetches: window in superchunk s, top-K-by-load of superchunk s-1 held resident ==")
print(f"{'N':>5}", end="")
for K in RESIDENT_KS:
    print(f"  {'K='+str(K):>12}", end="")
print("   (mean distinct MISSING experts = actual fetches)")
top_by = {}
for (l, s), m in mats.items():
    cnt = np.bincount(m.ravel(), minlength=E)
    top_by[(l, s)] = {K: set(np.argsort(-cnt)[:K].tolist()) for K in RESIDENT_KS}
for N in [1, 8, 64, 256, 2048]:
    print(f"{N:>5}", end="")
    for K in RESIDENT_KS:
        vals = []
        for s in full:
            if s == 0 or (s - 1) not in scs:
                continue
            for l in layers:
                m = mats[(l, s)]
                res = top_by[(l, s - 1)][K]
                nw = SC_TOKENS // N
                for w in range(nw):
                    u = set(np.unique(m[w * N:(w + 1) * N]).tolist())
                    vals.append(len(u - res))
        print(f"  {np.mean(vals):>12.1f}", end="")
    print()
