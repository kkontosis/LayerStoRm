"""Replicate the ENGINE's routed kFinal launch shape: cap=8 grid (topk rows,
256-expert offsets) with only `A_live` non-NULL experts (NULL-skip for the
rest) vs the tight cap=A_live grid the isolated bench used."""
import os
import sys
import numpy as np
import torch
DEPS = os.environ.get('LS_GEMM_DEPS',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'deps', 'LayerStoRmGemmKernels'))
sys.path.insert(0, DEPS)
import sm120_gemm_kernels as GK
from tests.test_gguf_grouped_gemm import GGUF_SPEC, make_gguf_weight, S_INT, F_MMVQ

WARMUP, ITERS, R = 30, 200, 16

def bench(fns):
    n = len(fns)
    for i in range(WARMUP): fns[i % n]()
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        ss[i].record(); fns[i % n](); es[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e)*1e3 for s, e in zip(ss, es)]))

print('GPU:', torch.cuda.get_device_name())
for label, name, N, K in [("gate/up Q4_K", "Q4_K", 2048, 6144), ("down Q5_K", "Q5_K", 6144, 2048)]:
    t = GGUF_SPEC[name]["t"]
    for live in (1, 2):
        # ENGINE-shape: 8 experts x 1 row, only `live` non-NULL b_ptrs -> cap=8
        A8 = (torch.randn(8, K, dtype=torch.float32)*0.4).to(torch.bfloat16).cuda()
        off8 = torch.arange(9, dtype=torch.int32).cuda()
        fns8 = []
        for r in range(R):
            ws = [make_gguf_weight(name, N, K, seed=r*991+e) for e in range(live)]
            ptrs = [w.data_ptr() for w in ws] + [0]*(8-live)
            b = torch.tensor(ptrs, dtype=torch.int64, device='cuda')
            fns8.append(lambda A_=A8, o=off8, b_=b, w_=ws: GK.gguf_grouped_gemm(A_, o, b_, t, S_INT, N, K, F_MMVQ))
        t8 = bench(fns8)
        # TIGHT-shape: cap=live, all non-NULL
        Al = (torch.randn(live, K, dtype=torch.float32)*0.4).to(torch.bfloat16).cuda()
        offl = torch.arange(live+1, dtype=torch.int32).cuda()
        fnsl = []
        for r in range(R):
            ws = [make_gguf_weight(name, N, K, seed=r*991+e) for e in range(live)]
            b = torch.tensor([w.data_ptr() for w in ws], dtype=torch.int64, device='cuda')
            fnsl.append(lambda A_=Al, o=offl, b_=b, w_=ws: GK.gguf_grouped_gemm(A_, o, b_, t, S_INT, N, K, F_MMVQ))
        tl = bench(fnsl)
        print(f'  {label:14s} live={live}: engine-shape cap=8+NULLs {t8:6.1f} us | tight cap={live} {tl:6.1f} us | dead-row tax {t8-tl:+5.1f}')
