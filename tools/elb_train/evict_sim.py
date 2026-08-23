"""Offline eviction-policy hit-rate simulator (TD-EPM-DRAFT-FEATURES Phase 1,
system-side PROXY on the honest held-out split).

The DSP52_BOOST "Oracle Belady-EVICTION" GO established that in the engine a
better eviction policy converts a hit-rate gain into wall (Belady +20.4pp
hit -> +3.0% wall; the realizable recurs<=16 bridge +10.2pp -> +4.5% wall).
This tool measures, on the held-out corpus committed DECODE demand trace,
the HIT RATE reachable by:

  LRU            recency eviction (the offline analog of the board's
                 recency term; the DSP52 baseline).
  freq           decayed-frequency + recency (the champion board score).
  belady         exact Belady (evict farthest next-use) — the ceiling.
  bridge16       recurs<=16 protection + LRU among unprotected (the
                 REALIZABLE oracle bridge the engine measured at +4.5%).
  pred_bridge    the TRAINED draft-hidden recur predictor drives the SAME
                 protect-then-LRU rule (protect iff model prob >= thr) —
                 the deployable predictor.  This is the offline judge of
                 "does the trained predictor capture the oracle bridge's
                 convertible hit gain?"
  prevunion      protect iff routed at previous committed position
                 (b0_prev temporal locality) — the free occurrence signal.

Cache model: a single pooled (layer, expert) cache of capacity C shared
across layers (POLICY_LAB finding #1: pooled Belady == partitioned 4-way
at the optimum; partitioning is free).  Demands are emitted in trajectory
order: for each committed decode position, for each MoE layer in ascending
id order, that layer's routed experts.  Capacity is swept; the regime of
interest is where LRU hit ~ the champion decode ~0.42 (DSP52_BOOST oracle
"Compulsory vs CAPACITY" table).
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np


def build_trace(pred_npz: dict, max_seqs: int | None = None):
    """Flatten the held-out prediction dump into a global demand trace in
    trajectory order.  Returns parallel arrays over demands:
      key  int64  (layer<<20)|expert   (unique cache key)
      prob f32    model recurrence probability at this demand
      prev int8   routed at previous committed position (occurrence)
      lab  int8   recur16 label
    plus `next_use` int64 (index of the same key's NEXT demand; N=never).
    """
    seq = pred_npz["seq"]; pos = pred_npz["pos"]; layer = pred_npz["layer"]
    exp = pred_npz["expert"].astype(np.int64)
    if max_seqs is not None:
        keep_seqs = sorted(set(seq.tolist()))[:max_seqs]
        m = np.isin(seq, keep_seqs)
        seq, pos, layer, exp = seq[m], pos[m], layer[m], exp[m]
        pred_npz = {k: (v[m] if k in ("prob", "occ_prev", "label",
                                      "nextdist") else v)
                    for k, v in pred_npz.items()}
    # trajectory order: by (seq, pos, layer).  Stable sort composes.
    order = np.lexsort((layer, pos, seq))
    layer_o = layer[order].astype(np.int64)
    key = (layer_o << 20) | (exp[order] & 0xFFFFF)
    prob = pred_npz["prob"][order]
    prev = pred_npz["occ_prev"][order]
    lab = pred_npz["label"][order]
    # nextdist = POSITION-distance to this key's next route (from the recur
    # sidecar): the Belady/bridge time axis is committed decode POSITIONS,
    # NOT flattened demand indices (a 16-token horizon spans ~16 positions,
    # each ~hundreds of per-layer demands).  -1 = never again in-sequence.
    nextdist = pred_npz["nextdist"][order].astype(np.int64)
    seq_o = seq[order]
    return {"key": key, "prob": prob, "prev": prev, "lab": lab,
            "nextdist": nextdist, "seq": seq_o}


def simulate(trace, cap: int, policy: str, thr: float = 0.5,
             freq_decay: float = 0.9048, freq_w: float = 60.0):
    """Run one policy over the trace; return hit_rate.  O(log cap) per
    demand via a lazily-invalidated min-heap of keep-scores (victim = the
    LOWEST keep-score resident).  Per-sequence cold cache (independent
    generations, no cross-sequence leak).

    Keep-score (higher = keep, victim = min):
      lru        recency tick.
      freq       recency tick + freq_w * decayed-freq (board analog).
      belady     next-use tick (nearer next-use = higher keep).
      bridge16   (protected?, recency): protected iff next-use within 16.
      pred_bridge(protected?, recency): protected iff model prob >= thr.
      prevunion  (protected?, recency): protected iff routed at prev pos.
    Two-tier policies encode protection as +BIG so protected keys always
    outrank unprotected; protected are only evicted when all resident are
    protected (no livelock), tie-broken by recency."""
    key = trace["key"]; nd = trace["nextdist"]
    prob = trace["prob"]; prev = trace["prev"]; seq = trace["seq"]
    N = len(key)
    BIG = float(N + 10)
    NEVER = float(N)                          # farthest possible next-use

    def keep_score(i, k):
        if policy == "lru":
            return float(i)
        if policy == "freq":
            return float(i) + freq_w * res_freq[k]
        if policy == "belady":
            # keep NEAREST next-use; never-again (nd<0) evicted first.
            d = float(nd[i]) if nd[i] >= 0 else NEVER
            return -d
        # two-tier protect-then-LRU (protection is a POSITION-horizon signal)
        if policy == "bridge16":
            p = 1 if (0 <= nd[i] <= 16) else 0
        elif policy == "pred_bridge":
            p = 1 if float(prob[i]) >= thr else 0
        elif policy == "prevunion":
            p = 1 if int(prev[i]) > 0 else 0
        else:
            raise ValueError(policy)
        return p * BIG * 2.0 + float(i)      # recency within each tier

    resident: set[int] = set()
    res_freq: dict[int, float] = {}
    ver: dict[int, int] = {}
    heap: list = []                          # (score, version, key)
    hits = 0
    cur_seq = -1
    for i in range(N):
        k = int(key[i])
        if int(seq[i]) != cur_seq:
            resident.clear(); res_freq.clear(); ver.clear(); heap.clear()
            cur_seq = int(seq[i])
        res_freq[k] = res_freq.get(k, 0.0) * freq_decay + 1.0
        if k in resident:
            hits += 1
        else:
            if len(resident) >= cap:
                while True:                  # pop stale entries lazily
                    _s, v, vk = heapq.heappop(heap)
                    if vk in resident and ver.get(vk) == v:
                        resident.discard(vk); res_freq.pop(vk, None)
                        ver.pop(vk, None)
                        break
            resident.add(k)
        v = ver.get(k, 0) + 1
        ver[k] = v
        heapq.heappush(heap, (keep_score(i, k), v, k))
    return hits / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions npz")
    ap.add_argument("--caps", default="200,400,700,1052,1500")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pred = dict(np.load(args.pred))
    trace = build_trace(pred, max_seqs=(args.max_seqs or None))
    N = len(trace["key"])
    base = float((trace["lab"] > 0).mean())
    print(f"trace: {N:,} decode demands, {len(set(trace['seq'].tolist()))} "
          f"seqs, base recur16 {base:.4f}")
    caps = [int(c) for c in args.caps.split(",")]
    policies = ["lru", "freq", "prevunion", "pred_bridge", "bridge16",
                "belady"]
    rows = []
    for cap in caps:
        r = {"cap": cap}
        for pol in policies:
            r[pol] = simulate(trace, cap, pol, thr=args.thr)
        rows.append(r)
        conv = ((r["pred_bridge"] - r["lru"]) /
                (r["belady"] - r["lru"]) if r["belady"] > r["lru"] else
                float("nan"))
        print(f"cap {cap:5d}  LRU {r['lru']:.4f}  freq {r['freq']:.4f}  "
              f"prevU {r['prevunion']:.4f}  PRED {r['pred_bridge']:.4f}  "
              f"bridge16 {r['bridge16']:.4f}  Belady {r['belady']:.4f}  "
              f"| pred captures {conv*100:.1f}% of Belady gain")
    if args.out:
        json.dump({"base_rate": base, "n_demands": N, "rows": rows,
                   "thr": args.thr}, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
