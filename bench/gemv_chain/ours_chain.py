#!/usr/bin/env python3
"""Mini decode FFN chain with OUR kernels (deps LayerStoRmGemmKernels), one
token (M=1), one routed expert, mirroring the engine's dispatch:
  o_proj (single-tensor mmvq, Q8_0)  -> +residual
  routed gate/up (grouped Q4_K, E=1) -> swiglu -> routed down (grouped Q5_K)
  shared gate/up (grouped Q8_0, E=1) -> swiglu -> shared down (grouped Q8_0)
  out = r + down + sh_down
Whole chain captured in one CUDA graph (like the engine FFN graph) and timed
per replay. Weights are REAL GLM-5.2 blk.10 bytes (see extract_weights.py).

Usage: CUDA_VISIBLE_DEVICES=<d> python3 ours_chain.py [--profile]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEPS = os.environ.get('LS_GEMM_DEPS',
                      os.path.join(HERE, '..', '..', 'deps', 'LayerStoRmGemmKernels'))
sys.path.insert(0, DEPS)
import sm120_gemm_kernels as GK

W = os.path.join(HERE, 'weights')
meta = json.load(open(os.path.join(W, 'meta.json')))
# deps GgufType ints (gguf_dequant_gemm.h): Q4_K=2 Q5_K=3 Q8_0=5
GGML2DEPS = {8: 5, 12: 2, 13: 3}
S_INT, F_MMVQ = 1, 1

WARMUP, ITERS = 30, 300


def load(name):
    m = meta[name]
    raw = np.fromfile(os.path.join(W, name + '.bin'), dtype=np.uint8)
    assert raw.nbytes == m['nbytes']
    return torch.from_numpy(raw).cuda(), GGML2DEPS[m['ggml_type']], m['n'], m['k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    torch.cuda.init()
    print('GPU:', torch.cuda.get_device_name())

    ws = {n: load(n) for n in ['o_proj', 'gate', 'up', 'down', 'sh_gate', 'sh_up', 'sh_down']}
    off1 = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    bptr = {n: torch.tensor([ws[n][0].data_ptr()], dtype=torch.int64, device='cuda')
            for n in ['gate', 'up', 'down', 'sh_gate', 'sh_up', 'sh_down']}

    x = torch.from_numpy(np.fromfile(os.path.join(W, 'x_f32.bin'), dtype=np.float32)) \
        .to(torch.bfloat16).cuda().view(1, 8192)
    res = torch.from_numpy(np.fromfile(os.path.join(W, 'res_f32.bin'), dtype=np.float32)) \
        .to(torch.bfloat16).cuda().view(1, 6144)

    def grouped(a, name):
        w, t, n, k = ws[name]
        return GK.gguf_grouped_gemm(a, off1, bptr[name], t, S_INT, n, k, F_MMVQ)

    def chain():
        w, t, n, k = ws['o_proj']
        o = GK.gguf_mmvq(x, w, t, n, k)                    # [1,6144]
        r = o + res
        g = grouped(r, 'gate')                             # [1,2048] Q4_K
        u = grouped(r, 'up')
        a = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
        d = grouped(a, 'down')                             # [1,6144] Q5_K
        sg = grouped(r, 'sh_gate')                         # Q8_0 (engine slow path)
        su = grouped(r, 'sh_up')
        sa = (torch.nn.functional.silu(sg.float()) * su.float()).to(torch.bfloat16)
        sd = grouped(sa, 'sh_down')
        return r + d + sd

    # warmup + output for the numerical comparison
    for _ in range(WARMUP):
        out = chain()
    torch.cuda.synchronize()
    out.float().cpu().numpy().tofile(os.path.join(HERE, 'out_ours.f32.bin'))

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                chain()
            torch.cuda.synchronize()
        for e in prof.key_averages():
            if e.device_type.name == 'CUDA' or 'kernel' in e.key.lower():
                print(f'  {e.key[:80]:80s} {e.self_device_time_total/10:9.1f} us/iter')

    # eager per-op breakdown (CUDA events around each op, median of ITERS)
    names = ['o_proj', 'add', 'gate', 'up', 'swiglu', 'down',
             'sh_gate', 'sh_up', 'sh_swiglu', 'sh_down', 'combine']
    evs = [[torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
           for _ in range(ITERS)]
    for i in range(ITERS):
        e = evs[i]
        w, t, n, k = ws['o_proj']
        e[0].record()
        o = GK.gguf_mmvq(x, w, t, n, k); e[1].record()
        r = o + res; e[2].record()
        g = grouped(r, 'gate'); e[3].record()
        u = grouped(r, 'up'); e[4].record()
        a = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16); e[5].record()
        d = grouped(a, 'down'); e[6].record()
        sg = grouped(r, 'sh_gate'); e[7].record()
        su = grouped(r, 'sh_up'); e[8].record()
        sa = (torch.nn.functional.silu(sg.float()) * su.float()).to(torch.bfloat16); e[9].record()
        sd = grouped(sa, 'sh_down'); e[10].record()
        out = r + d + sd; e[11].record()
    torch.cuda.synchronize()
    per = {nm: float(np.median([evs[i][j].elapsed_time(evs[i][j + 1]) * 1e3
                                for i in range(ITERS)]))
           for j, nm in enumerate(names)}
    tot_eager = float(np.median([evs[i][0].elapsed_time(evs[i][-1]) * 1e3
                                 for i in range(ITERS)]))

    # CUDA-graph capture of the whole chain (the engine-FFN-graph analog)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = chain()
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    tot_graph = float(np.median([s0.elapsed_time(e0) * 1e3
                                 for s0, e0 in zip(starts, ends)]))

    print('\nper-op eager (median us):')
    for nm in names:
        print(f'  {nm:10s} {per[nm]:8.1f}')
    print(f'chain eager  median: {tot_eager:8.1f} us')
    print(f'chain GRAPH  median: {tot_graph:8.1f} us')
    json.dump(dict(per_op=per, eager_us=tot_eager, graph_us=tot_graph,
                   gpu=torch.cuda.get_device_name()),
              open(os.path.join(HERE, f'ours_{os.environ.get("TAG", "run")}.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
