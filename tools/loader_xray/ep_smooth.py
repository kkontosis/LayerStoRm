#!/usr/bin/env python3
"""PHASE 1 — HYBRID-EP per-rank load-smoothing analytical diagnostic.

Reads an LS_PERF_TRACE CSV (champion regime). Reconstructs, per MoE command
(=one layer visit), the REAL fetched copies (misses) with their destination
EP-rank (GPU/link) and host source NUMA node. Then, per verify-CHUNK sweep,
computes:
  (a) per-layer per-rank fetch imbalance (max vs mean) + idle-rank fraction
  (b) per-rank smoothing CEILING: current Sigma_layer max_rank  vs the perfect-
      smoothing floor max_rank(Sigma_layer)  (link basis)
  (c) NODE-conservation cross-check: per-layer per-node load; the node hard
      floor max_node(Sigma_layer N_node) that per-rank smoothing CANNOT beat
      (single-homed slots => a copy loads its source node regardless of which
      rank/link pulls it). If node floor >= rank floor, node conservation
      defeats the smoothing.

Topology constants from FETCH_XRAY.md: per-link solo 47 GiB/s, per-node cap
64 GiB/s, copy = 26.34 MiB. GPU(pos)->node {0:2,1:3,2:0,3:2} (node2 shared by
g0+g3; node1 never local).
"""
import sys, csv, statistics
from collections import defaultdict

EXPERT_BYTES = 27623424           # 26.34 MiB
GIB = 2**30
T_LINK = EXPERT_BYTES / (47.0 * GIB) * 1e3   # ms per copy, one link solo
T_NODE = EXPERT_BYTES / (64.0 * GIB) * 1e3   # ms per copy at node cap
GPU_NODE = {0: 2, 1: 3, 2: 0, 3: 2}

def load(path):
    evs = []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            evs.append((int(row[0]), int(row[1]), int(row[2]),
                        int(row[3]), int(row[4]), int(row[5])))
    evs.sort(key=lambda e: e[0])
    return evs

def build(evs):
    moe = {}
    for ns, st, gpu, seq, key, tok in evs:
        if st == 12:   moe.setdefault(seq, {}).update(enter=ns, layer=key)
        elif st == 13: moe.setdefault(seq, {})['issued'] = ns
        elif st == 14: moe.setdefault(seq, {})['fin_enter'] = ns
        elif st == 15: moe.setdefault(seq, {})['fin_exit'] = ns
    # pair dispatch(4/6) with kDmaGpu(10) per (gpu,key)
    disp_q = defaultdict(list); xfers = []
    for ns, st, gpu, seq, key, tok in evs:
        if st in (4, 6):
            disp_q[(gpu, key)].append(ns)
        elif st == 10:
            q = disp_q.get((gpu, key))
            if q:
                disp = q.pop(0)
                xfers.append({'gpu': gpu, 'key': key, 'disp': disp,
                              'dur': tok, 'node': seq - 1})
    xfers.sort(key=lambda x: x['disp'])
    # per-link serialized timeline
    link_end = defaultdict(int)
    for x in xfers:
        s = max(x['disp'], link_end[x['gpu']])
        x['start'] = s; x['end'] = s + x['dur']; link_end[x['gpu']] = x['end']
    # assign to MoE windows [enter, fin_exit]
    import bisect
    wins = sorted((m['enter'], m.get('fin_exit', m['enter']), seq)
                  for seq, m in moe.items() if 'enter' in m)
    starts = [w[0] for w in wins]
    by_cmd = defaultdict(list)
    for x in xfers:
        i = bisect.bisect_right(starts, x['disp']) - 1
        if i >= 0 and x['disp'] <= wins[i][1]:
            by_cmd[wins[i][2]].append(x)
    # round segmentation (layer sweeps)
    cmds = sorted((m['enter'], seq, m.get('layer', -1))
                  for seq, m in moe.items() if 'enter' in m)
    rounds = []; cur = []; last_layer = 999
    for ent, seq, layer in cmds:
        if layer <= last_layer and cur and layer <= cur[0][1]:
            rounds.append(cur); cur = []
        cur.append((seq, layer, ent)); last_layer = layer
    if cur: rounds.append(cur)
    return moe, by_cmd, rounds

def analyze(tag, rounds, by_cmd, chunk_threshold):
    # classify each round
    chunk_rounds, decode_rounds = [], []
    for r in rounds:
        f = sum(len(by_cmd.get(seq, ())) for seq, _, _ in r)
        (chunk_rounds if f >= chunk_threshold else decode_rounds).append(r)
    print(f"\n{'='*74}\n{tag}: {len(rounds)} sweeps  "
          f"({len(chunk_rounds)} chunk >= {chunk_threshold} fetches, "
          f"{len(decode_rounds)} decode)\n{'='*74}")

    for label, rset in (("CHUNK", chunk_rounds), ("DECODE", decode_rounds)):
        if not rset: continue
        # accumulators
        sum_layer_max = 0.0     # Sigma_layer max_rank copies (link CP proxy, copies)
        sum_layer_mean = 0.0    # Sigma_layer mean_rank copies
        sum_layer_maxnode = 0.0 # Sigma_layer max_node copies
        tot_copies = 0
        n_layers = 0
        idle_slots = 0; total_slots = 0; idle_copies_avail = 0
        # per-sweep floors (link basis + node basis)
        cur_cp_ms = 0.0
        rank_floor_ms = 0.0
        node_floor_ms = 0.0
        node_floor_percmd_ms = 0.0   # Sigma_layer max_node*T_NODE (per-cmd node term)
        cur_cp_measured_ms = 0.0
        imb_ratios = []
        for r in rset:
            Tr = defaultdict(int)     # per-rank total copies this sweep
            Nnd = defaultdict(int)    # per-node total copies this sweep
            sweep_cur_cp = 0.0
            for seq, layer, _ in r:
                xs = by_cmd.get(seq, ())
                if not xs: continue
                n_layers += 1
                rc = defaultdict(int); nc = defaultdict(int)
                for x in xs:
                    rc[x['gpu']] += 1; nc[x['node']] += 1
                    Tr[x['gpu']] += 1; Nnd[x['node']] += 1
                c = len(xs); tot_copies += c
                mx = max(rc.values())
                # mean over the 4 ranks (idle ranks count as 0)
                mean = c / 4.0
                mxnode = max(nc.values())
                sum_layer_max += mx
                sum_layer_mean += mean
                sum_layer_maxnode += mxnode
                if mx > 0: imb_ratios.append(mx / mean if mean else 0)
                # idle-rank accounting: ranks below the layer max are idle for
                # (mx - rc[rank]) copy-slots after finishing
                for g in range(4):
                    total_slots += 1
                    deficit = mx - rc.get(g, 0)
                    if deficit > 0:
                        idle_slots += 1
                        idle_copies_avail += deficit
                # current per-cmd CP (LB model: max of link & node term)
                cp = max(mx * T_LINK, mxnode * T_NODE)
                cur_cp_ms += cp
                sweep_cur_cp += cp
                node_floor_percmd_ms += mxnode * T_NODE
                # measured span
                cur_cp_measured_ms += (max(x['end'] for x in xs)
                                       - min(x['disp'] for x in xs)) / 1e6
            # per-sweep perfect-smoothing floors
            if Tr:
                rank_floor_ms += max(Tr.values()) * T_LINK
                node_floor_ms += max(Nnd.values()) * T_NODE
        if n_layers == 0: continue
        print(f"\n-- {label} regime ({len(rset)} sweeps, {n_layers} layer-cmds, "
              f"{tot_copies} copies) --")
        print(f"  per-layer imbalance  max/mean ratio: "
              f"med {statistics.median(imb_ratios):.2f}  "
              f"mean {statistics.mean(imb_ratios):.2f}  "
              f"max {max(imb_ratios):.2f}")
        print(f"  idle-rank slots: {idle_slots}/{total_slots} "
              f"({100*idle_slots/total_slots:.0f}%)  "
              f"idle copy-slots available for future work: {idle_copies_avail}")
        print(f"  COUNTS:  Sigma_layer max_rank = {sum_layer_max:.0f}   "
              f"Sigma_layer mean_rank = {sum_layer_mean:.0f}   "
              f"Sigma_layer max_node = {sum_layer_maxnode:.0f}")
        # (b) per-rank smoothing ceiling (link basis, copy counts -> ms)
        print(f"\n  (b) PER-RANK SMOOTHING CEILING (link basis):")
        print(f"    current  Sigma_layer max_rank*T_link      = {cur_cp_ms:8.1f} ms "
              f"(measured span {cur_cp_measured_ms:8.1f} ms)")
        print(f"    rank floor  max_rank(Sigma_layer)*T_link  = {rank_floor_ms:8.1f} ms "
              f"(perfect per-rank smoothing, node-free)")
        ceil_rankonly = 1 - rank_floor_ms / cur_cp_ms if cur_cp_ms else 0
        print(f"    -> rank-only ceiling (ignoring nodes)      = {100*ceil_rankonly:5.1f}%")
        # (c) node conservation
        print(f"\n  (c) NODE-CONSERVATION cross-check:")
        print(f"    node hard floor  max_node(Sigma_layer)*T_node = {node_floor_ms:8.1f} ms "
              f"(INVARIANT to rank smoothing)")
        print(f"    per-cmd node term  Sigma_layer max_node*T_node = {node_floor_percmd_ms:8.1f} ms")
        # persistent-bottleneck breakdown: per-rank / per-node total share, and
        # how often each rank/node is THE per-layer max (stability of bottleneck)
        rank_tot = defaultdict(int); node_tot = defaultdict(int)
        rank_ismax = defaultdict(int); node_ismax = defaultdict(int); nmax=0
        for r in rset:
            for seq, layer, _ in r:
                xs = by_cmd.get(seq, ())
                if not xs: continue
                rc = defaultdict(int); nc = defaultdict(int)
                for x in xs:
                    rc[x['gpu']] += 1; nc[x['node']] += 1
                    rank_tot[x['gpu']] += 1; node_tot[x['node']] += 1
                mxr = max(rc.values()); mxn = max(nc.values()); nmax += 1
                for g,v in rc.items():
                    if v == mxr: rank_ismax[g] += 1
                for nd,v in nc.items():
                    if v == mxn: node_ismax[nd] += 1
        tc = sum(rank_tot.values())
        print(f"  per-RANK total share: " + "  ".join(
            f"g{g}={100*rank_tot[g]/tc:.0f}%(max@{100*rank_ismax[g]/nmax:.0f}%)"
            for g in sorted(rank_tot)))
        print(f"  per-NODE total share: " + "  ".join(
            f"n{nd}={100*node_tot[nd]/tc:.0f}%(max@{100*node_ismax[nd]/nmax:.0f}%)"
            for nd in sorted(node_tot)))
        achievable_floor = max(rank_floor_ms, node_floor_ms)
        ceil_real = 1 - achievable_floor / cur_cp_ms if cur_cp_ms else 0
        binding = "NODE" if node_floor_ms >= rank_floor_ms else "RANK(link)"
        print(f"    achievable floor = max(rank,node)          = {achievable_floor:8.1f} ms "
              f"[binding: {binding}]")
        print(f"    ==> REAL per-rank smoothing ceiling        = {100*ceil_real:5.1f}%  "
              f"of fetch-phase")
        # translate to e2e: fetch phase is a fraction of wall
        node_defeats = node_floor_ms >= cur_cp_ms * 0.97
        print(f"    node-conservation defeats smoothing? "
              f"{'YES' if node_defeats else 'NO'} "
              f"(node floor {100*node_floor_ms/cur_cp_ms:.0f}% of current CP)")

def main():
    path = sys.argv[1]
    chunk_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    evs = load(path)
    print(f"file={path}  events={len(evs)}")
    moe, by_cmd, rounds = build(evs)
    print(f"moe_cmds={len(moe)}  rounds={len(rounds)}  "
          f"T_link={T_LINK:.4f}ms/copy  T_node={T_NODE:.4f}ms/copy")
    # fetch-count histogram per sweep
    hist = defaultdict(int)
    for r in rounds:
        hist[sum(len(by_cmd.get(s, ())) for s, _, _ in r)] += 1
    print("sweep fetch-count histogram (count: n_sweeps):")
    for k in sorted(hist):
        print(f"   {k:5d}: {hist[k]}")
    analyze(path.split('/')[-1], rounds, by_cmd, chunk_threshold)

if __name__ == "__main__":
    main()
