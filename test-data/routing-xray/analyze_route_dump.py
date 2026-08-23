#!/usr/bin/env python3
# Analyze LS_DRIFT_DUMP routing records from a superchunk prefill run.
#
# Record layout (little-endian), one per (gating call = sub-chunk, layer, gpu):
#   int32 hdr[6] = {seq, layer_idx, gpu, num_tokens, n_experts, topk}
#   float logits[num_tokens*n_experts]      (skipped)
#   int32 idx   [num_tokens*topk]
#   float w     [num_tokens*topk]
#
# Outputs:
#  A) Hot-set stability: Jaccard of top-50 (by token count) expert sets between
#     consecutive superchunks, per layer; plus sc0-vs-scN long-range overlap.
#  B) Gate-threshold coverage: per full 2048-token superchunk, distinct experts
#     with >=1 assignment whose per-token NORMALIZED gate weight >= tau, for
#     tau in {0 (baseline), 0.02, 0.05, 0.08, 0.10}.
import struct, sys
from collections import defaultdict

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/route_dump_sc.bin"
SC_TOKENS = 2048
TOP_HOT = 50
TAUS = [0.02, 0.05, 0.08, 0.10]

recs = 0
first_gpu = None
gpus_seen = set()
n_experts_g = topk_g = None
# per (layer, sc): counts per expert (assignments), and per-tau kept-expert sets
counts = defaultdict(lambda: None)          # (layer, sc) -> np.zeros(E)
kept = defaultdict(lambda: defaultdict(set))  # (layer, sc) -> tau -> set(expert)
mass_dropped = defaultdict(lambda: defaultdict(float))  # weight mass dropped per tau
mass_total = defaultdict(float)
tok_off = defaultdict(int)                  # layer -> token offset so far (dedup'd gpu)

with open(PATH, "rb") as f:
    while True:
        hdr = f.read(24)
        if len(hdr) < 24:
            break
        seq, layer, gpu, nt, ne, tk = struct.unpack("<6i", hdr)
        if not (0 < nt <= 4096 and 0 < ne <= 1024 and 0 < tk <= 16):
            print(f"corrupt header at seq={seq}: nt={nt} ne={ne} tk={tk}", file=sys.stderr)
            break
        n_experts_g, topk_g = ne, tk
        f.seek(nt * ne * 4, 1)  # skip logits
        idx = np.frombuffer(f.read(nt * tk * 4), dtype="<i4").reshape(nt, tk)
        w = np.frombuffer(f.read(nt * tk * 4), dtype="<f4").reshape(nt, tk)
        recs += 1
        gpus_seen.add(gpu)
        if first_gpu is None:
            first_gpu = gpu
        if gpu != first_gpu:
            continue  # replicated gating — only the first rank contributes
        sc = tok_off[layer] // SC_TOKENS
        tok_off[layer] += nt
        k = (layer, sc)
        if counts[k] is None:
            counts[k] = np.zeros(ne, dtype=np.int64)
        np.add.at(counts[k], idx.ravel(), 1)
        # per-token normalization (removes routed_scaling_factor)
        s = w.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        wn = w / s
        mass_total[k] += float(w.sum())
        for tau in TAUS:
            keep_mask = wn >= tau
            kept[k][tau].update(np.unique(idx[keep_mask]).tolist())
            mass_dropped[k][tau] += float(w[~keep_mask].sum())

layers = sorted({l for (l, _) in counts})
scs = sorted({s for (_, s) in counts})
full_scs = [s for s in scs if all(counts.get((l, s)) is not None and counts[(l, s)].sum() >= SC_TOKENS * topk_g for l in layers)]
print(f"records={recs} gpus={sorted(gpus_seen)} (analyzed gpu {first_gpu}) "
      f"E={n_experts_g} topk={topk_g} layers={len(layers)} superchunks={scs} full={full_scs}")

def top_set(c, n=TOP_HOT):
    return set(np.argsort(-c)[:n].tolist())

# A) Jaccard stability
print("\n== A) Hot-set stability: Jaccard(top-50 by token count) ==")
rand_j = (TOP_HOT * TOP_HOT / n_experts_g) / (2 * TOP_HOT - TOP_HOT * TOP_HOT / n_experts_g)
print(f"random-baseline Jaccard for 50-of-{n_experts_g}: {rand_j:.3f}")
for a, b in zip(scs, scs[1:]):
    js = []
    for l in layers:
        ca, cb = counts.get((l, a)), counts.get((l, b))
        if ca is None or cb is None:
            continue
        sa, sb = top_set(ca), top_set(cb)
        js.append(len(sa & sb) / len(sa | sb))
    js = np.array(js)
    print(f"sc{a}->sc{b}: mean {js.mean():.3f}  median {np.median(js):.3f}  "
          f"min {js.min():.3f}  max {js.max():.3f}  (n={len(js)} layers)")
# long-range: first vs last full
if len(scs) >= 3:
    a, b = scs[0], scs[-1]
    js = np.array([len(top_set(counts[(l, a)]) & top_set(counts[(l, b)])) /
                   len(top_set(counts[(l, a)]) | top_set(counts[(l, b)]))
                   for l in layers if counts.get((l, a)) is not None and counts.get((l, b)) is not None])
    print(f"long-range sc{a}->sc{b}: mean {js.mean():.3f}  median {np.median(js):.3f}  "
          f"min {js.min():.3f}  max {js.max():.3f}")

# per-layer profile (worst / best stability)
mean_j_by_layer = {}
for l in layers:
    js = []
    for a, b in zip(scs, scs[1:]):
        ca, cb = counts.get((l, a)), counts.get((l, b))
        if ca is None or cb is None:
            continue
        js.append(len(top_set(ca) & top_set(cb)) / len(top_set(ca) | top_set(cb)))
    if js:
        mean_j_by_layer[l] = float(np.mean(js))
ordered = sorted(mean_j_by_layer.items(), key=lambda kv: kv[1])
print("least-stable layers:", ", ".join(f"L{l}={j:.2f}" for l, j in ordered[:5]))
print("most-stable layers: ", ", ".join(f"L{l}={j:.2f}" for l, j in ordered[-5:]))

# B) threshold coverage on FULL 2048-token superchunks
print(f"\n== B) Fetched experts per (layer, 2048-token superchunk) after gate-weight cut ==")
base_cov, tau_cov, tau_mass = [], {t: [] for t in TAUS}, {t: [] for t in TAUS}
for s in full_scs:
    for l in layers:
        c = counts.get((l, s))
        if c is None:
            continue
        base_cov.append(int((c > 0).sum()))
        for t in TAUS:
            tau_cov[t].append(len(kept[(l, s)][t]))
            tau_mass[t].append(mass_dropped[(l, s)][t] / max(mass_total[(l, s)], 1e-9))
base = np.array(base_cov)
print(f"baseline (tau=0):   mean {base.mean():6.1f} / {n_experts_g}  ({100*base.mean()/n_experts_g:.1f}%)   range {base.min()}-{base.max()}")
for t in TAUS:
    cv = np.array(tau_cov[t]); ms = np.array(tau_mass[t])
    print(f"tau={t:.2f}: fetched mean {cv.mean():6.1f} / {n_experts_g}  ({100*cv.mean()/n_experts_g:.1f}%)   "
          f"range {cv.min()}-{cv.max()}   bytes-saved {100*(1-cv.mean()/base.mean()):.1f}%   "
          f"gate-mass dropped {100*ms.mean():.2f}%")
