#!/usr/bin/env python3
"""Q1/Q2: quantify the baseline-vs-ACT routing divergence magnitude + cascade.

Reads the committed oracles (engine ground truth). Reports per-(token,layer)
expert-set overlap, the first divergence, and the cascade over tokens.
"""
import csv
import statistics
from collections import defaultdict, Counter


def load(f):
    with open(f) as fh:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def sets(rows):
    d = defaultdict(set)
    for r in rows:
        d[(r['token'], r['layer'])].add(r['expert_idx'])
    return d


b = load('assets/oracle_baseline.csv')
a = load('assets/oracle_act.csv')
t = load('assets/routed_trace.csv')
sb, sa, st = sets(b), sets(a), sets(t)
keys = sorted(sb)

print("=== Q1: routing divergence magnitude (|base ∩ act| / 8) ===")
print("baseline expert-set == routed_trace expert-set:",
      all(sb[k] == st[k] for k in keys), "(routed_trace tracks BASELINE)")
ov = [len(sb[k] & sa[k]) for k in keys]
print("(token,layer)s total      :", len(keys))
print("set-INEQUAL count         :", sum(1 for k in keys if sb[k] != sa[k]))
print("overlap mean/median/min/max: %.3f / %d / %d / %d"
      % (statistics.mean(ov), statistics.median(ov), min(ov), max(ov)))
print("overlap histogram (#shared experts of 8):",
      dict(sorted(Counter(ov).items())))
print("random-routing baseline (E[|A∩B|] for two K=8 of 256): %.3f" % (8 * 8 / 256))

print("\n=== Q2: first divergence (FP-drift signature) ===")
for k in keys:
    if sb[k] != sa[k]:
        print("first divergence (token,layer):", k)
        print("  baseline:", sorted(sb[k]))
        print("  act     :", sorted(sa[k]))
        print("  shared:", len(sb[k] & sa[k]), "/ 8  (single near-tie flip:",
              sorted(sb[k] - sa[k]), "->", sorted(sa[k] - sb[k]), ")")
        break
t0 = [len(sb[(0, L)] & sa[(0, L)]) for L in range(3, 61)]
print("  token 0: layers 3..21 all overlap 8 (bit-identical):",
      all(x == 8 for x in t0[:19]))
print("  token 0 mean overlap: %.3f (drift confined to a few late layers)"
      % (sum(t0) / len(t0)))

print("\n=== Q1: cascade over tokens (autoregressive amplification) ===")
print("tok | mean expert-overlap/8")
for tok in [0, 1, 2, 3, 4, 5, 6, 8, 10, 20, 40, 60, 80, 99]:
    ovs = [len(sb[(tok, L)] & sa[(tok, L)]) for L in range(3, 61)]
    print("%3d | %.3f" % (tok, sum(ovs) / len(ovs)))
