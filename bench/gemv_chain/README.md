# gemv_chain — isolated 1-token / 1-expert decode FFN chain, ours vs llama.cpp

Chains the per-layer decode GEMVs exactly as the engine dispatches them
(M=1, one routed expert, real GLM-5.2 blk.10 packed weights):

    o_proj (Q8_0 6144x8192, single-tensor mmvq)  -> +residual
    routed gate/up (Q4_K 2048x6144, grouped E=1) -> swiglu -> down (Q5_K 6144x2048)
    shared gate/up/down (Q8_0, grouped E=1 — the engine's shared-expert path)
    out = r + down + sh_down

and runs the identical chain (same weight bytes, same input) through ggml's
CUDA backend (llama.cpp mmvq + its automatic CUDA-graph capture). One pass
streams ~116 MB of weights, so L2 self-evicts — realistic cold reads.

## Run
1. `python3 extract_weights.py`   (needs llama.cpp/gguf-py on PYTHONPATH; writes weights/)
2. ours:  `CUDA_VISIBLE_DEVICES=<d> python3 ours_chain.py`  (deps .so on sys.path; torch)
3. ggml:  build llama.cpp with GGML_CUDA=ON (arch 120), then
   `g++ -O2 -std=c++17 -I <llama.cpp>/ggml/include ggml_chain.cpp -o ggml_chain \
      -L <llama.cpp>/build/bin -lggml -lggml-base -lggml-cuda -Wl,-rpath,<llama.cpp>/build/bin`
   `CUDA_VISIBLE_DEVICES=<d> ./ggml_chain weights`
4. compare `out_ours.f32.bin` vs `out_ggml.f32.bin` (cosine).

## Results (2026-07-13, llama.cpp a410713, deps a855dfc, engine 36f7ad59)
Whole chain, CUDA-graphed, median us/iter; cosine(ours, ggml) = 0.999967:

| GPU  | ours  | ggml  | ratio |
|------|------:|------:|------:|
| 5090 | 318.8 |  94.7 | 3.37x |
| 5080 | 380.2 | 151.6 | 2.51x |

Per-GEMV (5090, ours = eager CUDA events / ggml = nsys per-launch):

| GEMV               | ours us | ggml us | ratio | note |
|--------------------|--------:|--------:|------:|------|
| o_proj Q8_0 53.5MB |   42.3  |  33.1   | 1.3x  | near parity (our single-tensor port) |
| gate/up Q4_K 7.1MB |   20.5  |   6.0   | 3.4x  | grouped compact+cpasync E=1 under-fills (256 CTAs) |
| down Q5_K 8.7MB    |   16.4  |   6.9   | 2.4x  | |
| sh gate/up Q8_0 13.4MB | 96.3 |  9.5   | 10x   | grouped Q8_0 pipe path (cp.async-ineligible) |
| sh down Q8_0 13.4MB|   40.5  |   9.3   | 4.4x  | |

ggml runs every shape at the DRAM floor (see handoff §11b/§12). The torch
swiglu/add glue (~35 us total) is a harness artifact — the engine's fused
swiglu is ~1 us; subtract it when reading the chain totals.

## E=1 fast path (LS_GGUF_E1_KSPLIT=1, deps 9b48062 — default OFF)
Re-run with the fast path enabled (`LS_GGUF_E1_KSPLIT=1 python3 ours_chain.py`):
shared Q8_0 GEMVs 96.3/96.3/40.5 -> 20.5 us each (5090); whole chain
318.8 -> 155.3 us (5090), 380.2 -> 239.2 us (5080); cosine vs ggml unchanged
(0.999967). In-engine verdict + why it defaults OFF: handoff §12c.
