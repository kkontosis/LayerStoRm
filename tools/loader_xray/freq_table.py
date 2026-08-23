#!/usr/bin/env python3
"""Derive a per-(layer,expert) fetch-frequency table from LS_PERF_TRACE dumps
(Wave-2 M3 arena host placement, src/core/memory/arena_placement.h).

Counts DISPATCHED expert H2D copies (stages 4=kEnqueueInline /
6=kDispatchStaged; `key` = layer<<16|expert) per key, EXCLUDING the prefill
sweeps and the first 3 warm-up sweeps — the same round segmentation
fetch_xray.py uses (layer sweeps split on layer wrap; prefill = leading
rounds with >3000 fetches). The result ranks keys by DEMAND-FETCH frequency,
which is exactly the quantity the placement policy optimizes (bytes on the
wire per key), not routing frequency (hot-in-VRAM experts route often but
fetch rarely).

Usage:
  freq_table.py OUT.csv TRACE.csv [TRACE2.csv ...] [--prefill-thr N]

Multiple traces accumulate (e.g. one plain-decode arm + one champion
chunk-verify arm → a table representative of both regimes). Output CSV:
"layer,expert,count" (+ a provenance comment header), consumable by
LS_ARENA_PLACE_FREQ.

Also prints the concentration profile (share of counted fetches covered by
the top-K keys) so the HBM-capacity coverage is visible before a run.
"""
import csv
import sys
from collections import defaultdict


def load(path):
    evs = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            evs.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                        int(row[4]), int(row[5])))
    evs.sort(key=lambda e: e[0])
    return evs


def count_one(path, prefill_thr):
    evs = load(path)

    # MoE command windows (12=kMoeEnter carries layer; 15=kMoeFinalizeExit).
    moe = {}
    for ns, st, gpu, seq, key, tok in evs:
        if st == 12:
            moe.setdefault(seq, {}).update(enter=ns, layer=key)
        elif st == 15:
            moe.setdefault(seq, {})['fin_exit'] = ns

    # Dispatched transfers (stage 4/6) with their dispatch ns + key.
    xfers = [(ns, key) for ns, st, gpu, seq, key, tok in evs if st in (4, 6)]

    # Assign each transfer to the MoE window containing its dispatch
    # (same reconstruction as fetch_xray.py).
    import bisect
    wins = sorted((m['enter'], m.get('fin_exit', m['enter']), seq)
                  for seq, m in moe.items() if 'enter' in m)
    starts = [w[0] for w in wins]
    per_cmd = defaultdict(list)
    for ns, key in xfers:
        i = bisect.bisect_right(starts, ns) - 1
        if i >= 0 and ns <= wins[i][1]:
            per_cmd[wins[i][2]].append(key)

    # Round segmentation: layer sweeps (split when the layer wraps).
    cmds = sorted((m['enter'], seq, m.get('layer', -1))
                  for seq, m in moe.items() if 'enter' in m)
    rounds, cur, last_layer = [], [], 999
    for ent, seq, layer in cmds:
        if layer <= last_layer and cur and layer <= cur[0][1]:
            rounds.append(cur)
            cur = []
        cur.append((seq, layer, ent))
        last_layer = layer
    if cur:
        rounds.append(cur)

    def round_fetches(r):
        return sum(len(per_cmd.get(seq, ())) for seq, _, _ in r)

    # Exclude leading prefill rounds (huge unions) + 3 warm-up sweeps.
    n_prefill = 0
    for r in rounds:
        if round_fetches(r) > prefill_thr:
            n_prefill += 1
        else:
            break
    body = rounds[n_prefill + 3:]

    counts = defaultdict(int)
    body_fetches = 0
    for r in body:
        for seq, _, _ in r:
            for key in per_cmd.get(seq, ()):
                counts[key] += 1
                body_fetches += 1
    print(f"{path}: {len(xfers)} dispatches, {len(rounds)} rounds "
          f"({n_prefill} prefill + 3 warmup excluded) -> "
          f"{body_fetches} counted fetches, {len(counts)} distinct keys")
    return counts


def main():
    args = sys.argv[1:]
    prefill_thr = 3000
    if '--prefill-thr' in args:
        i = args.index('--prefill-thr')
        prefill_thr = int(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit(__doc__)
    out_path, traces = args[0], args[1:]

    total = defaultdict(int)
    for t in traces:
        for k, c in count_one(t, prefill_thr).items():
            total[k] += c

    rows = sorted(((k >> 16, k & 0xFFFF, c) for k, c in total.items()),
                  key=lambda r: (-r[2], r[0], r[1]))
    grand = sum(c for _, _, c in rows)
    with open(out_path, 'w') as f:
        f.write("# per-(layer,expert) demand-fetch counts "
                "(tools/loader_xray/freq_table.py)\n")
        f.write(f"# sources: {' '.join(traces)}\n")
        f.write(f"# total counted fetches: {grand}\n")
        for layer, expert, c in rows:
            f.write(f"{layer},{expert},{c}\n")
    print(f"wrote {out_path}: {len(rows)} keys, {grand} fetches")

    # Concentration profile: coverage of top-K keys.
    csum = 0
    marks = {}
    for i, (_, _, c) in enumerate(rows, 1):
        csum += c
        for K in (500, 1000, 2000, 2500, 3000, 4000, 8000, 16000):
            if i == K:
                marks[K] = csum / grand
    print("top-K fetch coverage: " + "  ".join(
        f"K={k}:{v * 100:.1f}%" for k, v in sorted(marks.items())))


if __name__ == '__main__':
    main()
