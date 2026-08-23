#!/usr/bin/env python3
"""RARE-on-NODE-1 CPU-offload residency A/B — FIXED-trajectory replay.

QUESTION (coordinator, 2026-08-11). The engine's (e) TARGET-NODE RARE offload arm
(enode_e01_r1.log: node=1 rarity_thresh=0.1) collapsed the loader hit_rate
0.3677 → 0.2035 (~45%). But that arm ran Q8 forced-CPU numerics (LS_CPU_EXPERT_
LOSSLESS=0) which FORK the speculative trajectory (committed 107→110, acc
0.482→0.522, lookups 42966→45050 — the two runs are NOT the same token stream).
So the raw hit drop CONFLATES two effects: (1) a REAL residency effect of
excluding the rare/node-1 set from the GPU cache, and (2) Q8 trajectory-fork
noise. This sim SEPARATES them by replaying ONE fixed trajectory (the champion
trace) through BOTH arms with an IDENTICAL cache model:

  A = champion    : every routed expert is cache-simulated on its target GPU.
                    Calibrate caps so A reproduces the champion anchor 0.3677.
  B = offload set : the exact engine rule — EXCLUDE from the GPU cache (host-
                    served, never cached, not a GPU lookup) every (layer,expert)
                    that is BOTH (i) RARE (M3 freq/fhot < rarity_thresh) AND
                    (ii) home-noded on the GPU-free node 1. Same fixed trace.

Because A and B share ONE trajectory, any hit delta is PURE residency — no fork.

NODE-1 MODELLING (honest caveat). Node 1 is GPU-FREE: NumaManager.expert_home_
node round-robins only over the GPU-attached nodes {0,2,3}, so node 1 NEVER
appears as a round-robin home and the trace's `bank` field confirms it (banks
{0,2,3} only). The engine's node-1 experts are an ARENA CROSS-NODE SPILL artifact
(PinnedExpertArena filled node 1 with ~4407 of the ~17.6k DDR slots ≈ 25%), whose
exact per-(layer,expert) membership is NOT recoverable from the trace. We model
it two honest ways and report all:
  * rarity-ONLY (node filter OFF): exclude ALL rare experts — the residency UPPER
    BOUND. Node-independent ⇒ the decisive residency answer. If this HOLDS near
    0.3677 the node-1 subset is trivially safe (fork noise); if it DROPS, rare
    experts are cache-resident (real residency effect).
  * node-1 proxy: rare AND a deterministic ~25% hash slice (node 1 = 1 of 4 DDR
    nodes) — the engine's actual offloaded population size. Reported for scale.
  * node-1-only (rarity filter OFF): the 25% slice alone, to isolate which filter
    (rarity vs node) drives any drop.

INSTRUMENTATION: for every excluded set we report |set| (unique layer,expert),
how many of those were actually cache-HITS in A (the ones whose exclusion costs
hit), and the resulting hit + delta.
"""
import argparse
import gzip
import json
import sys
from collections import OrderedDict, defaultdict

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
FREQ_TABLE = os.path.join(_ROOT, "test-data/placement/glm52_fetch_freq_m3.csv")
TRACE = os.path.join(_HERE, "assets/traj_4gpu_canonical.jsonl.gz")
M = 4  # GPUs


class Lru:
    __slots__ = ("cap", "od")

    def __init__(self, cap):
        self.cap = cap
        self.od = OrderedDict()

    def hit(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return True
        return False

    def insert(self, key):
        self.od[key] = 1
        self.od.move_to_end(key)
        while len(self.od) > self.cap:
            self.od.popitem(last=False)

    def evict(self, key):
        self.od.pop(key, None)


def load_freq(path):
    freq = {}
    mx = 0
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        L, e, c = line.split(",")
        L, e, c = int(L), int(e), int(c)
        freq[(L, e)] = c
        if c > mx:
            mx = c
    return freq, mx


def load_trace(path):
    """Return by_layer: dict layer_idx -> list over tokens of [(expert_idx, j), ...]."""
    f = gzip.open(path, "rt")
    # token index inferred by grouping: the trace is token-major, NLAYERS records
    # per token in ascending layer_idx. Detect token boundaries by layer_idx reset.
    recs = [json.loads(line) for line in f]
    layers = sorted({r["layer_idx"] for r in recs})
    nlayers = len(layers)
    ntok = len(recs) // nlayers
    lpos = {L: i for i, L in enumerate(layers)}
    by_layer = {L: [None] * ntok for L in layers}
    for i, r in enumerate(recs):
        t = i // nlayers
        L = r["layer_idx"]
        if t < ntok:
            by_layer[L][t] = [(e["expert_idx"], e["j"]) for e in r["experts"]]
    return by_layer, layers, ntok


def run(by_layer, layers, ntok, caps, B, excluded=None):
    """Token-by-token per-GPU LRU replay. `excluded` = set of (layer,expert) that are
    host-served (evicted, excluded from GPU cache & lookups). Returns metrics + the
    per-key A-hit accounting is done by the caller via excl_hits when excluded given."""
    if excluded is None:
        excluded = set()
    caches = [Lru(caps[g]) for g in range(M)]
    lookups = hits = 0
    import math
    nchunks = math.ceil(ntok / B)
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        for L in layers:
            for t in range(t0, t1):
                for (e, j) in by_layer[L][t]:
                    key = (L, e)
                    g = j if 0 <= j < M else (e % M)
                    if key in excluded:
                        caches[g].evict(key)
                        continue
                    lookups += 1
                    if caches[g].hit(key):
                        hits += 1
                    else:
                        caches[g].insert(key)
    return {"lookups": lookups, "hits": hits,
            "hit": hits / max(lookups, 1), "misses": lookups - hits}


def a_hit_per_key(by_layer, layers, ntok, caps, B):
    """Mode-A replay recording, per (layer,expert), #lookups and #hits — so we can
    report how many of an excluded set were actually HITS in the champion (the uses
    whose exclusion forfeits a real cache hit)."""
    import math
    caches = [Lru(caps[g]) for g in range(M)]
    look = defaultdict(int)
    hit = defaultdict(int)
    nchunks = math.ceil(ntok / B)
    for c in range(nchunks):
        t0 = c * B
        t1 = min(t0 + B, ntok)
        for L in layers:
            for t in range(t0, t1):
                for (e, j) in by_layer[L][t]:
                    key = (L, e)
                    g = j if 0 <= j < M else (e % M)
                    look[key] += 1
                    if caches[g].hit(key):
                        hit[key] += 1
                    else:
                        caches[g].insert(key)
    return look, hit


def det_node1(e, keep_frac=0.25):
    """Deterministic pseudo-random ~keep_frac slice of expert ids, proxy for the
    arena spill onto GPU-free node 1 (1 of 4 DDR banks ≈ 25%). Hash on expert_idx
    (a whole expert spills to one node, layer-independent, matching arena homing)."""
    h = (e * 2654435761) & 0xFFFFFFFF
    return (h % 1000) < int(keep_frac * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--freq", default=FREQ_TABLE)
    ap.add_argument("--caps", default=None, help="per-GPU slots c,c,c,c")
    ap.add_argument("--cap", type=int, default=None, help="uniform per-GPU slots")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--rarity", type=float, default=0.1)
    ap.add_argument("--node1-frac", type=float, default=0.25)
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep uniform caps to find the 0.3677 anchor, then stop")
    args = ap.parse_args()

    freq, fhot = load_freq(args.freq)
    by_layer, layers, ntok = load_trace(args.trace)
    thr = args.rarity * fhot
    print(f"trace {args.trace}: {ntok} tokens, {len(layers)} layers "
          f"(layer_idx {min(layers)}..{max(layers)}), B={args.B}")
    print(f"freq table: {len(freq)} keys, fhot(max)={fhot}, "
          f"rarity_thresh={args.rarity} ⇒ rare ⇔ freq < {thr:g}")

    if args.calibrate:
        print("\nCALIBRATION — uniform caps vs mode-A hit (target champion 0.3677):")
        for cap in (24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 200, 256, 305):
            r = run(by_layer, layers, ntok, [cap] * M, args.B)
            print(f"  cap={cap:4d}  hit={r['hit']:.4f}  lookups={r['lookups']}  "
                  f"misses={r['misses']}")
        return

    if args.caps:
        caps = [int(x) for x in args.caps.split(",")]
    elif args.cap:
        caps = [args.cap] * M
    else:
        caps = [305] * M

    # ── mode A (champion) ───────────────────────────────────────────────────
    A = run(by_layer, layers, ntok, caps, args.B)
    look, hit = a_hit_per_key(by_layer, layers, ntok, caps, args.B)
    print(f"\ncaps={caps}")
    print(f"A (champion, all-GPU):  hit={A['hit']:.4f}  lookups={A['lookups']}  "
          f"misses={A['misses']}   [anchor target 0.3677]")

    # ── build exclusion sets over the (layer,expert) that OCCUR in the trace ──
    occ_keys = set(look.keys())
    rare_keys = {(L, e) for (L, e) in occ_keys if freq.get((L, e), 0) < thr}
    node1_keys = {(L, e) for (L, e) in occ_keys if det_node1(e, args.node1_frac)}
    rare_node1 = rare_keys & node1_keys

    def report(name, excl):
        # A-side accounting for the excluded set
        set_lookups = sum(look[k] for k in excl)
        set_hits = sum(hit[k] for k in excl)
        B = run(by_layer, layers, ntok, caps, args.B, excluded=excl)
        d = B["hit"] - A["hit"]
        dpct = 100.0 * d / A["hit"]
        print(f"\n[{name}]")
        print(f"  excluded keys (unique layer,expert) : {len(excl)}")
        print(f"  their A-side uses / A-side HITS      : {set_lookups} / {set_hits}"
              f"  (A hit-rate on the set = {set_hits/max(set_lookups,1):.3f})")
        print(f"  B hit={B['hit']:.4f}  lookups={B['lookups']}  misses={B['misses']}")
        print(f"  Δhit vs A = {d:+.4f} ({dpct:+.1f}%)   "
              f"[engine e01 was 0.3677→0.2035 = -44.7%]")
        return B

    report("B_rarity  (rare only, NO node filter — residency UPPER BOUND)", rare_keys)
    report(f"B_node1   (rare AND node-1 ~{int(args.node1_frac*100)}% spill proxy — "
           f"engine e01 set)", rare_node1)
    report(f"B_node1only (node-1 ~{int(args.node1_frac*100)}% slice, NO rarity — "
           f"isolates the node filter)", node1_keys)


if __name__ == "__main__":
    main()
