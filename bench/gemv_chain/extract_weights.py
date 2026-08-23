#!/usr/bin/env python3
"""Extract real GLM-5.2 blk.10 decode-GEMV weights (packed GGUF bytes) for the
mini FFN-chain benchmark. o_proj is sliced to the per-rank TP half (K=8192);
routed expert = expert index 7. Also generates the fixed input/residual
activations (bf16-rounded, saved as f32 so both pipelines consume identical
values)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llama.cpp', 'gguf-py'))
from gguf import GGUFReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(OUT, exist_ok=True)
GGUF_DIR = os.environ.get('LS_GGUF_DIR',
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..', 'test-data', 'GLM-5.2-GGUF-Q4_K_XL'))
EXPERT = 7

want = {
    'blk.10.attn_output.weight': 'o_proj',
    'blk.10.ffn_gate_exps.weight': 'gate',
    'blk.10.ffn_up_exps.weight': 'up',
    'blk.10.ffn_down_exps.weight': 'down',
    'blk.10.ffn_gate_shexp.weight': 'sh_gate',
    'blk.10.ffn_up_shexp.weight': 'sh_up',
    'blk.10.ffn_down_shexp.weight': 'sh_down',
}
meta = {}
import glob
for f in sorted(glob.glob(GGUF_DIR + '/*.gguf')):
    r = GGUFReader(f)
    for t in r.tensors:
        key = want.get(t.name)
        if not key:
            continue
        data = t.data
        if key == 'o_proj':
            # (6144 rows, 17408 B) K=16384 -> per-rank K=8192 = first 8704 B/row
            data = np.ascontiguousarray(data[:, :8704])
            n, k, ttype = 6144, 8192, 8
        elif key in ('gate', 'up'):
            data = np.ascontiguousarray(data[EXPERT])   # (2048, 3456)
            n, k, ttype = 2048, 6144, 12
        elif key == 'down':
            data = np.ascontiguousarray(data[EXPERT])   # (6144, 1408)
            n, k, ttype = 6144, 2048, 13
        elif key in ('sh_gate', 'sh_up'):
            n, k, ttype = 2048, 6144, 8
        else:  # sh_down
            n, k, ttype = 6144, 2048, 8
        raw = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        raw.tofile(os.path.join(OUT, key + '.bin'))
        meta[key] = dict(n=n, k=k, ggml_type=ttype, nbytes=int(raw.nbytes), src=t.name)
        print(key, meta[key])

assert len(meta) == 7, f'missing tensors: {set(want.values()) - set(meta)}'

# Fixed activations, bf16-rounded then stored as f32 (identical on both sides).
rng = np.random.default_rng(52)
def bf16_round(a):
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x8000) & 0xFFFF0000).view(np.float32)
x = bf16_round(rng.standard_normal(8192).astype(np.float32) * 0.4)
res = bf16_round(rng.standard_normal(6144).astype(np.float32) * 0.4)
x.tofile(os.path.join(OUT, 'x_f32.bin'))
res.tofile(os.path.join(OUT, 'res_f32.bin'))
meta['_input'] = dict(x_len=8192, res_len=6144, seed=52)
with open(os.path.join(OUT, 'meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=1)
print('done ->', OUT)
