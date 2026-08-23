#!/usr/bin/env python3
"""FETCH_XRAY stage-1 analyzer (spec/reports/FETCH_XRAY.md, 2026-08-04): per-link H2D DMA occupancy timeline from the
LS_PERF_TRACE CSV (ns,stage,gpu,cmd_seq,key,token).

Reconstruction model: each GPU has ONE h2d stream; copies execute serially in
dispatch order. start_i = max(dispatch_ns_i, end_{i-1}); end_i = start_i + dur_i
(dur from kDmaGpu GPU timing events). Gaps = start_i - end_{i-1} when the stream
went idle. Host-steady-clock vs GPU-clock skew does not accumulate: every idle
gap resets the chain to a host timestamp.

Stages (perf_trace.h): 4=kEnqueueInline 5=kEnqueueStaged 6=kDispatchStaged
7=kDetected 10=kDmaGpu(token=dur_ns, cmd_seq=src_numa+1) 12=kMoeEnter
13=kMoeFetchIssued 14=kMoeFinalizeEnter 15=kMoeFinalizeExit
16=kExpertArrived(cmd_seq=moe cmd).
"""
import sys, csv, statistics
from collections import defaultdict

EXPERT_BYTES = None  # set from --bytes
GPU_NODE = {0: 2, 1: 3, 2: 0, 3: 2}  # config position -> local NUMA node (arena_map dump)

def load(path):
    evs = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            evs.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                        int(row[4]), int(row[5])))
    return evs

def main():
    trace = sys.argv[1]
    global EXPERT_BYTES
    EXPERT_BYTES = int(sys.argv[2])
    evs = load(trace)
    evs.sort(key=lambda e: e[0])

    # ── MoE command windows ──────────────────────────────────────────────
    moe = {}  # cmd_seq -> dict
    for ns, st, gpu, seq, key, tok in evs:
        if st == 12:
            moe.setdefault(seq, {}).update(enter=ns, layer=key)
        elif st == 13:
            moe.setdefault(seq, {})['issued'] = ns
        elif st == 14:
            moe.setdefault(seq, {})['fin_enter'] = ns
        elif st == 15:
            moe.setdefault(seq, {})['fin_exit'] = ns
        elif st == 16:
            m = moe.setdefault(seq, {})
            m.setdefault('arrivals', []).append((ns, gpu, key))

    # ── transfers: pair dispatch (4/6) with kDmaGpu (10) per (gpu,key) ──
    disp_q = defaultdict(list)   # (gpu,key) -> [dispatch records]
    staged_at = {}               # token -> stage-5 ns
    xfers = []                   # dicts
    for ns, st, gpu, seq, key, tok in evs:
        if st == 5:
            staged_at[tok] = ns
        elif st in (4, 6):
            disp_q[(gpu, key)].append({'disp': ns, 'tok': tok, 'stage': st,
                                       'staged': staged_at.pop(tok, None)})
        elif st == 10:
            q = disp_q.get((gpu, key))
            if q:
                d = q.pop(0)
                xfers.append({'gpu': gpu, 'key': key, 'disp': d['disp'],
                              'dur': tok, 'src_numa': seq - 1,
                              'staged': d['staged'], 'inline': d['stage'] == 4})
        elif st == 7:
            q = disp_q.get((gpu, key))  # kDetected for a transfer w/o kDmaGpu
            # ignore; kDmaGpu handles the pairing

    xfers.sort(key=lambda x: x['disp'])

    # ── per-link serialized timeline ─────────────────────────────────────
    link_end = defaultdict(int)
    for x in xfers:
        s = max(x['disp'], link_end[x['gpu']])
        x['start'] = s
        x['end'] = s + x['dur']
        x['gap'] = max(0, x['disp'] - link_end[x['gpu']]) if link_end[x['gpu']] else 0
        link_end[x['gpu']] = x['end']

    # ── assign transfers to MoE windows ─────────────────────────────────
    wins = sorted((m['enter'], m.get('fin_exit', m['enter']), seq)
                  for seq, m in moe.items() if 'enter' in m)
    import bisect
    starts = [w[0] for w in wins]
    for x in xfers:
        i = bisect.bisect_right(starts, x['disp']) - 1
        x['cmd'] = None
        if i >= 0:
            w = wins[i]
            if x['disp'] <= w[1]:
                x['cmd'] = w[2]

    by_cmd = defaultdict(list)
    for x in xfers:
        if x['cmd'] is not None:
            by_cmd[x['cmd']].append(x)

    # ── round segmentation: layer sweeps ────────────────────────────────
    cmds = sorted((m['enter'], seq, m.get('layer', -1)) for seq, m in moe.items()
                  if 'enter' in m)
    rounds = []   # list of [ (seq, layer, enter) ]
    cur = []
    last_layer = 999
    for ent, seq, layer in cmds:
        if layer <= last_layer and cur and layer <= cur[0][1]:
            rounds.append(cur); cur = []
        cur.append((seq, layer, ent))
        last_layer = layer
    if cur: rounds.append(cur)

    # classify rounds by fetch volume
    def round_fetches(r):
        return sum(len(by_cmd.get(seq, ())) for seq, _, _ in r)

    print(f"total events={len(evs)}  transfers_paired={len(xfers)}  "
          f"moe_cmds={len(moe)}  rounds={len(rounds)}")
    print(f"expert_bytes={EXPERT_BYTES} ({EXPERT_BYTES/2**20:.2f} MiB)")

    # round summary table
    print("\n── rounds (fetch count per layer-sweep) ──")
    hist = defaultdict(int)
    for r in rounds:
        hist[round_fetches(r)] += 1
    for k in sorted(hist):
        print(f"  fetches={k:5d}  rounds={hist[k]}")

    # ── per-regime analysis helper ──────────────────────────────────────
    def analyze(tag, seqs):
        X = [x for seq in seqs for x in by_cmd.get(seq, ())]
        if not X:
            print(f"\n== {tag}: no transfers =="); return
        X.sort(key=lambda x: x['start'])
        print(f"\n==== {tag}: {len(X)} transfers, {len(seqs)} moe cmds ====")
        tot_bytes = len(X) * EXPERT_BYTES

        # fetch-phase wall: union of [first disp, last end] per command
        span = 0
        for seq in seqs:
            xs = by_cmd.get(seq)
            if not xs: continue
            span += max(x['end'] for x in xs) - min(x['disp'] for x in xs)
        agg = tot_bytes / span * 1e9 / 2**30 if span else 0
        print(f"fetch bytes={tot_bytes/2**30:.2f} GiB  Σper-cmd fetch span="
              f"{span/1e6:.1f} ms  AGGREGATE={agg:.1f} GiB/s (per-cmd-span basis)")

        # per-link stats
        print(f"{'gpu':>3} {'n':>6} {'GiB':>7} {'busy_ms':>9} {'eff_GB/s':>9} "
              f"{'gap_ms':>8} {'gapN':>6} {'gap_med_us':>10} {'gap_p90_us':>10} "
              f"{'stg%':>5} {'stg_lat_med_us':>14}")
        for g in sorted({x['gpu'] for x in X}):
            gx = [x for x in X if x['gpu'] == g]
            busy = sum(x['dur'] for x in gx)
            eff = len(gx) * EXPERT_BYTES / busy * 1e9 / 2**30 if busy else 0
            # gaps within this command set only (consecutive on the same link)
            gaps = []
            gxs = sorted(gx, key=lambda x: x['start'])
            for a, b in zip(gxs, gxs[1:]):
                if b['disp'] > a['end']:
                    gaps.append((b['disp'] - a['end']) / 1e3)
            stg = [x for x in gx if x['staged'] is not None]
            stg_lat = [ (x['disp'] - x['staged'])/1e3 for x in stg ]
            print(f"{g:>3} {len(gx):>6} {len(gx)*EXPERT_BYTES/2**30:>7.2f} "
                  f"{busy/1e6:>9.1f} {eff:>9.1f} "
                  f"{sum(gaps)/1e3:>8.1f} {len(gaps):>6} "
                  f"{statistics.median(gaps) if gaps else 0:>10.0f} "
                  f"{(sorted(gaps)[int(len(gaps)*0.9)] if gaps else 0):>10.0f} "
                  f"{100*len(stg)/len(gx):>5.0f} "
                  f"{statistics.median(stg_lat) if stg_lat else 0:>14.0f}")

        # NUMA source distribution per destination gpu
        print("src NUMA node distribution per dst gpu (rows=gpu, cols=node):")
        nodes = sorted({x['src_numa'] for x in X})
        print("     " + "".join(f"n{n:<7}" for n in nodes) + "local%")
        for g in sorted({x['gpu'] for x in X}):
            gx = [x for x in X if x['gpu'] == g]
            cnt = defaultdict(int)
            for x in gx: cnt[x['src_numa']] += 1
            loc = 100 * cnt.get(GPU_NODE.get(g, -9), 0) / len(gx)
            print(f"  g{g}: " + "".join(f"{cnt.get(n,0):<8}" for n in nodes)
                  + f"{loc:.0f}%")
        # per-src-node effective bandwidth (while busy)
        print("per src node: n, xfers, eff GB/s while busy (median per-xfer):")
        for n in nodes:
            nx = [x for x in X if x['src_numa'] == n]
            meds = statistics.median([EXPERT_BYTES / x['dur'] * 1e9 / 2**30 for x in nx])
            print(f"  n{n}: {len(nx):>6}  med {meds:.1f} GiB/s/xfer")

        # source-node ganging: per ns of link-busy time, how many links read
        # the same node concurrently (sampled at transfer starts)
        edges = []
        for x in X:
            edges.append((x['start'], 1, x['src_numa'], x['gpu']))
            edges.append((x['end'], -1, x['src_numa'], x['gpu']))
        edges.sort()
        active = defaultdict(int)  # node -> count
        last_t = None
        node_time = defaultdict(lambda: defaultdict(int))  # node -> conc -> ns
        for t, d, n, g in edges:
            if last_t is not None:
                for nn, c in active.items():
                    if c > 0: node_time[nn][c] += t - last_t
            active[n] += d
            last_t = t
        print("src-node concurrency (ns share of that node's busy time at k simultaneous links):")
        for n in sorted(node_time):
            tot = sum(node_time[n].values())
            row = "  ".join(f"k={k}:{100*v/tot:.0f}%" for k, v in sorted(node_time[n].items()))
            print(f"  n{n}: total {tot/1e6:>8.1f} ms   {row}")

        # concurrent-vs-solo per-transfer bandwidth on same src node
        solo, shared = [], []
        # mark each transfer: did another link read the same node during it?
        by_node = defaultdict(list)
        for x in X: by_node[x['src_numa']].append(x)
        for n, xs in by_node.items():
            xs2 = sorted(xs, key=lambda x: x['start'])
            for i, x in enumerate(xs2):
                ov = any(o is not x and o['start'] < x['end'] and o['end'] > x['start']
                         and o['gpu'] != x['gpu'] for o in xs2[max(0,i-8):i+8])
                bw = EXPERT_BYTES / x['dur'] * 1e9 / 2**30
                (shared if ov else solo).append(bw)
        if solo and shared:
            print(f"per-xfer link GiB/s: SOLO-on-node med {statistics.median(solo):.1f} "
                  f"(n={len(solo)})  SHARED-node med {statistics.median(shared):.1f} "
                  f"(n={len(shared)})")

        # ── placement x-ray (Wave-2 M3): node-gang histogram + topology LB ──
        # Per-command topology lower bound (FETCH_XRAY.md §5):
        #   LB = Σ_cmd max(max-per-link-copies·B/LINK, max-per-DDR-node·B/NODE)
        # HBM-node copies (nodes ≥4 on this box) are EXEMPT from the 64-GiB/s
        # DDR term (HBM read bw ≫ link rate) but still count via the link term.
        LINK = 47 * 2**30
        NODE = 64 * 2**30
        HBM_NODES = {4, 5, 6, 7}
        lb_exempt = 0.0   # HBM copies exempt from the node term
        lb_all = 0.0      # every node 64-capped (the FETCH_XRAY.md §5 LB)
        gang_hist = defaultdict(int)      # max-per-DDR-node copies per cmd
        hbm_copies = sum(1 for x in X if x['src_numa'] in HBM_NODES)
        for seq in seqs:
            xs = by_cmd.get(seq)
            if not xs: continue
            per_l = defaultdict(int); per_n = defaultdict(int)
            per_n_all = defaultdict(int)
            for x in xs:
                per_l[x['gpu']] += 1
                per_n_all[x['src_numa']] += 1
                if x['src_numa'] not in HBM_NODES:
                    per_n[x['src_numa']] += 1
            max_n = max(per_n.values()) if per_n else 0
            gang_hist[max_n] += 1
            lterm = max(per_l.values()) * EXPERT_BYTES / LINK
            lb_exempt += max(lterm, max_n * EXPERT_BYTES / NODE) * 1e9
            lb_all += max(lterm,
                          max(per_n_all.values()) * EXPERT_BYTES / NODE) * 1e9
        print(f"placement x-ray: HBM-sourced copies {hbm_copies}/{len(X)} "
              f"({100*hbm_copies/len(X):.1f}%)  LB(all-node-64) "
              f"{lb_all/1e6:.0f} ms (actual/LB {span/max(lb_all,1):.2f}x)  "
              f"LB(HBM-exempt) {lb_exempt/1e6:.0f} ms (actual/LB "
              f"{span/max(lb_exempt,1):.2f}x)")
        print("  per-cmd max-copies-on-one-DDR-node histogram: " + "  ".join(
            f"{k}:{v}" for k, v in sorted(gang_hist.items())))

        # ── per-command decomposition: span vs ideal ────────────────────
        # T_actual = span; T_imb = max_g bytes_g / SOLO_RATE (imbalance-limited
        # ideal at solo per-link rate); T_bal = total/(4*SOLO_RATE).
        SOLO = 46.8 * 2**30  # measured solo-on-node per-link rate (B/s)
        t_act = t_imb = t_bal = 0.0
        in_span_idle = 0.0   # link involved in cmd, idle inside its own busy window
        in_span_busy = 0.0
        gaps_in_cmd = []
        for seq in seqs:
            xs = by_cmd.get(seq)
            if not xs: continue
            s0 = min(x['disp'] for x in xs); s1 = max(x['end'] for x in xs)
            t_act += (s1 - s0)
            per_g = defaultdict(int)
            for x in xs: per_g[x['gpu']] += 1
            t_imb += max(per_g.values()) * EXPERT_BYTES / SOLO * 1e9
            t_bal += len(xs) * EXPERT_BYTES / (4 * SOLO) * 1e9
            for g, cntg in per_g.items():
                gx = sorted((x for x in xs if x['gpu'] == g),
                            key=lambda x: x['start'])
                busy = sum(x['dur'] for x in gx)
                in_span_busy += busy
                # idle from this link's first dispatch to its last end
                lspan = gx[-1]['end'] - gx[0]['disp']
                in_span_idle += max(0, lspan - busy)
                for a, b in zip(gx, gx[1:]):
                    if b['disp'] > a['end']:
                        gaps_in_cmd.append((b['disp'] - a['end']) / 1e3)
        print(f"per-cmd decomposition: T_actual {t_act/1e6:.0f} ms | "
              f"T_imbalance-ideal {t_imb/1e6:.0f} ms | T_balanced-ideal "
              f"{t_bal/1e6:.0f} ms -> gap vs bal {t_act/max(t_bal,1):.2f}x, "
              f"imbalance share {(t_imb-t_bal)/max(t_act-t_bal,1)*100:.0f}%")
        print(f"within-cmd per-link: busy {in_span_busy/1e6:.0f} ms, idle-inside-"
              f"link-window {in_span_idle/1e6:.0f} ms "
              f"({100*in_span_idle/max(in_span_busy+in_span_idle,1):.0f}% idle); "
              f"within-cmd gapN={len(gaps_in_cmd)} med "
              f"{statistics.median(gaps_in_cmd) if gaps_in_cmd else 0:.0f} us "
              f"sum {sum(gaps_in_cmd)/1e3:.1f} ms")

        # issue-side: dispatch spread within each command
        spreads, first_lat = [], []
        for seq in seqs:
            xs = by_cmd.get(seq)
            if not xs or seq not in moe or 'enter' not in moe[seq]: continue
            ent = moe[seq]['enter']
            ds = sorted(x['disp'] for x in xs)
            first_lat.append((ds[0] - ent) / 1e3)
            spreads.append((ds[-1] - ds[0]) / 1e3)
        if spreads:
            print(f"issue-side per cmd: enter→first-dispatch med "
                  f"{statistics.median(first_lat):.0f} us (p90 "
                  f"{sorted(first_lat)[int(len(first_lat)*0.9)]:.0f}); "
                  f"first→last-dispatch spread med {statistics.median(spreads):.0f} us "
                  f"(p90 {sorted(spreads)[int(len(spreads)*0.9)]:.0f})")
        # per-cmd wall decomposition: enter→issued, issued→fin_enter, fin
        w1, w2, w3 = [], [], []
        for seq in seqs:
            m = moe.get(seq, {})
            if all(k in m for k in ('enter','issued','fin_enter','fin_exit')):
                w1.append((m['issued']-m['enter'])/1e6)
                w2.append((m['fin_enter']-m['issued'])/1e6)
                w3.append((m['fin_exit']-m['fin_enter'])/1e6)
        if w1:
            print(f"cmd walls ms (med): enter→issued {statistics.median(w1):.2f}  "
                  f"issued→fin_enter {statistics.median(w2):.2f}  "
                  f"finalize {statistics.median(w3):.2f}  (n={len(w1)})")

    # ── regime split from round classification ──────────────────────────
    # heuristics printed; actual thresholds chosen by inspection
    import json
    if len(sys.argv) > 3 and sys.argv[3] == '--dump-rounds':
        for i, r in enumerate(rounds):
            print(i, len(r), round_fetches(r), r[0][1], r[-1][1])
        return

    # regimes: prefill = leading sweeps with huge unions; warmup = next 3;
    # then threshold splits decode vs chunk sweeps.
    thr = int(sys.argv[3]) if len(sys.argv) > 3 else 450
    n_prefill = 0
    for r in rounds:
        if round_fetches(r) > 3000: n_prefill += 1
        else: break
    body = rounds[n_prefill + 3:]
    pre_seqs = [seq for r in rounds[:n_prefill] for seq, _, _ in r]
    dec_seqs = [seq for r in body if 0 < round_fetches(r) <= thr
                for seq, _, _ in r]
    chk_seqs = [seq for r in body if round_fetches(r) > thr
                for seq, _, _ in r]
    n_dec = sum(1 for r in body if 0 < round_fetches(r) <= thr)
    n_chk = sum(1 for r in body if round_fetches(r) > thr)
    print(f"\nregimes: prefill sweeps={n_prefill}, warmup skipped=3, "
          f"decode sweeps={n_dec}, chunk sweeps={n_chk} (thr={thr})")
    analyze(f"DECODE sweeps (fetches<={thr})", dec_seqs)
    analyze(f"CHUNK sweeps (fetches>{thr})", chk_seqs)
    analyze("PREFILL sweeps", pre_seqs)

if __name__ == '__main__':
    main()
