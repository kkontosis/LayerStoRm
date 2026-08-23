#!/usr/bin/env python3
"""Multi-expert GEMV comparison at the routed decode shapes (M_e=1 per expert):

  variants per E in {1,2,3,4,8}:
    grouped   — the engine's grouped kernel, ONE launch, E experts
                (env-dependent: default = compact+cpasync; set
                 LS_GGUF_GROUPED_KSPLIT_COMPACT=1 for the k-split (N,E) grid)
    seq       — E separate single-tensor mmvq launches, one stream (back-to-back)
    conc      — E separate single-tensor mmvq launches, E concurrent streams

  Shapes: routed gate/up Q4_K 2048x6144 and down Q5_K 6144x2048.
  R weight rotation sets keep reads DRAM-cold-ish (R*E*bytes >> L2 where possible).
  Reports eager (CUDA-event) median and CUDA-graph replay median.
"""
import os
import sys

import argparse

import numpy as np
import torch

DEPS = os.environ.get('LS_GEMM_DEPS',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'deps', 'LayerStoRmGemmKernels'))
sys.path.insert(0, DEPS)
import sm120_gemm_kernels as GK
from tests.test_gguf_grouped_gemm import GGUF_SPEC, make_gguf_weight, build_group, S_INT, F_MMVQ

WARMUP, ITERS = 30, 200
ES = [1, 2, 3, 4, 8]
SHAPES = [("gate/up Q4_K", "Q4_K", 2048, 6144), ("down    Q5_K", "Q5_K", 6144, 2048)]


def bench(fns):
    n = len(fns)
    for i in range(WARMUP):
        fns[i % n]()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        fns[i % n]()
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]))


def bench_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception:
        return float('nan')
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=1, help='rows per expert (batch B)')
    args = ap.parse_args()
    M = args.m
    torch.manual_seed(0)
    dev_name = torch.cuda.get_device_name()
    ksplit = os.environ.get('LS_GGUF_GROUPED_KSPLIT_COMPACT') == '1'
    print(f'GPU: {dev_name}  grouped-variant: {"ksplit(N,E)" if ksplit else "cpasync(N/8,E)"}  M(rows/expert)={M}')
    l2 = torch.cuda.get_device_properties(0).L2_cache_size / 1048576.0
    print(f'L2 = {l2:.0f} MB')

    for label, name, N, K in SHAPES:
        t = GGUF_SPEC[name]["t"]
        wbytes = N * (K // GGUF_SPEC[name]["qk"]) * GGUF_SPEC[name]["bytes"]
        print(f'\n== {label}  N={N} K={K}  ({wbytes/1e6:.2f} MB/expert) ==')
        print(f'{"E":>2} {"grouped":>18} {"seq-single":>18} {"conc-single":>18}  (eager us / graph us; per-expert in parens)')
        for E in ES:
            # rotation count: keep total footprint <= ~1.5 GB
            R = max(2, min(16, int(1.5e9 / (wbytes * E))))
            A = (torch.randn(max(2, E) * M, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
            A1 = A[:M].contiguous()

            # grouped: R groups of E experts (M_e=M each)
            groups = [build_group(name, [M] * E, N, K, seed=r * 131 + E) for r in range(R)]
            g_fns = [
                (lambda A_=A_, off=off, b=b: GK.gguf_grouped_gemm(A_, off, b, t, S_INT, N, K, F_MMVQ))
                for (A_, off, b, _w) in groups
            ]
            g_e = bench(g_fns)
            g_g = bench_graph(g_fns[0])

            # weights for the single-tensor variants (R sets of E weights)
            wsets = [[make_gguf_weight(name, N, K, seed=r * 977 + e) for e in range(E)]
                     for r in range(R)]

            def seq_fn(ws):
                def f():
                    for w in ws:
                        GK.gguf_mmvq(A1, w, t, N, K)
                return f
            s_e = bench([seq_fn(ws) for ws in wsets])
            s_g = bench_graph(seq_fn(wsets[0]))

            streams = [torch.cuda.Stream() for _ in range(E)]
            def conc_fn(ws):
                def f():
                    cur = torch.cuda.current_stream()
                    for st, w in zip(streams, ws):
                        st.wait_stream(cur)
                        with torch.cuda.stream(st):
                            GK.gguf_mmvq(A1, w, t, N, K)
                    for st in streams:
                        cur.wait_stream(st)
                return f
            c_e = bench([conc_fn(ws) for ws in wsets])
            c_g = bench_graph(conc_fn(wsets[0]))

            def fmt(e, g):
                pe = g / E if g == g else e / E
                return f'{e:6.1f}/{g:6.1f} ({pe:5.1f})'
            print(f'{E:>2} {fmt(g_e, g_g):>18} {fmt(s_e, s_g):>18} {fmt(c_e, c_g):>18}   R={R}')


if __name__ == '__main__':
    main()
