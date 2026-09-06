#!/usr/bin/env python
"""Localize run-to-run served-prefill divergence from an LS_DRIFT_DUMP file
(TD-SERVE-PREFILL-NONDET-RUN-TO-RUN).

LS_DRIFT_DUMP=<path> makes the engine append one binary record per
(gating call) — attention emit_gating on the FAR path, moe self-gating
otherwise — for EVERY request of the boot:

  int32 hdr[6] = {seq, layer, gpu, num_tokens, n_experts, topk}
  float logits[nt*ne]; int32 idx[nt*topk]; float w[nt*topk]

This tool segments the record stream into requests by finding the maximal
runs of prefill-shaped records (num_tokens >= 32) of length >= min_run
(default 1000): request k = records from prefill-run k's start to prefill-
run k+1's start (its decode records included).  It then aligns two chosen
requests record-by-record and reports the FIRST record whose logits or
top-K indices differ, with magnitude and position — bracketing the source
between that record's (chunk, layer) gating and the previous aligned one.

Usage: drift_dump_diff.py DUMP [A B] [--min-run N]
  A, B: request ordinals to compare (default 1 2 — i.e. the two probes
  after a warm-up request 0).

Both runs must come from the SAME boot with the dump active (the dump's
D2H+sync serialization is part of the arm; compare dump-on vs dump-on).
"""
import argparse
import struct

import numpy as np


def scan(path):
    metas, offs = [], []
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            hdr = f.read(24)
            if len(hdr) < 24:
                break
            s, layer, gpu, nt, ne, topk = struct.unpack("<6i", hdr)
            body = 4 * nt * ne + 8 * nt * topk
            f.seek(body, 1)
            if f.tell() > off + 24 + body:  # short file guard
                break
            metas.append((layer, gpu, nt, ne, topk))
            offs.append(off)
    return metas, offs


def segment(metas, min_run):
    runs, i = [], 0
    while i < len(metas):
        if metas[i][2] >= 32:
            j = i
            while j < len(metas) and metas[j][2] >= 32:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    blocks = []
    for k, (s, _) in enumerate(runs):
        e = runs[k + 1][0] if k + 1 < len(runs) else len(metas)
        blocks.append((s, e))
    return blocks


def read_rec(f, off):
    f.seek(off)
    s, layer, gpu, nt, ne, topk = struct.unpack("<6i", f.read(24))
    logits = np.frombuffer(f.read(4 * nt * ne), dtype="<f4")
    idx = np.frombuffer(f.read(4 * nt * topk), dtype="<i4")
    return (layer, gpu, nt, ne, topk), logits, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("a", nargs="?", type=int, default=1)
    ap.add_argument("b", nargs="?", type=int, default=2)
    ap.add_argument("--min-run", type=int, default=1000)
    args = ap.parse_args()

    metas, offs = scan(args.dump)
    blocks = segment(metas, args.min_run)
    print(f"{len(metas)} records, {len(blocks)} request blocks "
          f"(sizes {[e - s for s, e in blocks]})")
    if max(args.a, args.b) >= len(blocks):
        raise SystemExit("request ordinal out of range")
    (a0, a1), (b0, b1) = blocks[args.a], blocks[args.b]
    n = min(a1 - a0, b1 - b0)
    f = open(args.dump, "rb")
    first, ndiff, nshape = None, 0, 0
    for k in range(n):
        ma, mb = metas[a0 + k], metas[b0 + k]
        if ma != mb:
            nshape += 1
            if nshape == 1:
                print(f"first SHAPE mismatch at aligned record {k}: "
                      f"{ma} vs {mb} (speculative decode shapes may "
                      f"legitimately differ once tokens diverge)")
            continue
        _, la, ia = read_rec(f, offs[a0 + k])
        _, lb, ib = read_rec(f, offs[b0 + k])
        leq = np.array_equal(la, lb)
        ieq = np.array_equal(ia, ib)
        if leq and ieq:
            continue
        ndiff += 1
        if first is None:
            first = k
            layer, gpu, nt, ne, topk = ma
            d = np.abs(la - lb)
            j = int(np.argmax(d))
            print(f"FIRST DIVERGENCE at aligned record {k}: layer={layer} "
                  f"gpu={gpu} nt={nt}")
            print(f"  max|dlogit|={d.max():.3e} at (token {j // ne}, "
                  f"expert {j % ne}): {la[j]:.8f} vs {lb[j]:.8f}; "
                  f"rows with any diff="
                  f"{int((d.reshape(nt, ne) > 0).any(axis=1).sum())}/{nt}; "
                  f"topk_idx_equal={ieq}")
    if first is None:
        print(f"requests {args.a} and {args.b}: BIT-IDENTICAL over {n} "
              f"aligned records ({nshape} shape mismatches skipped)")
    else:
        print(f"requests {args.a} and {args.b}: {ndiff}/{n} records differ, "
              f"first at {first} ({nshape} shape mismatches)")


if __name__ == "__main__":
    main()
