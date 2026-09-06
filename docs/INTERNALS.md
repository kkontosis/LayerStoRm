# Engine internals

Bold names describe what each part does; an italic parenthetical attributes
the published design it follows (see [REFERENCES.md](REFERENCES.md)) or
marks work that originated here. The implementations are ours unless
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) records adapted code.

- **FP8 compressed-latent MLA** *(SnapMLA-like)* — near-lossless SM120-native attention kernels: the compressed latent in FP8, the RoPE slice kept BF16 (644 B/token/layer — 44% below BF16).
- **4-bit KV quantization** *(TurboQuant-like)* — very efficient KV-cache compression behind the same attention interface (386 B/token/layer, ~66% below BF16); composes with the other KV codecs per tier.
- **NUMA-aware host arena** — per-node pinned expert pools sized per bank, including CPU-less HBM-as-RAM nodes; `cudaHostRegister`-pinned (no `RLIMIT_MEMLOCK` ceiling), io_uring O_DIRECT preload, THP-backed registration.
- **PagedAttention** — paged KV with copy-on-write forks and metadata-only promotion of accepted speculative tokens.
- **KV tiering** *(HiSparse-like)* — sparse-attention-guided hot-VRAM / pinned-host KV hierarchy, so long contexts don't have to fit in VRAM.
- **TP with KV sharding** — decode context parallelism: each attention GPU holds a disjoint 1/tp of the sequence.
- **TP with weight sharding** — attention/dense weights split across the TP pair.
- **Expert parallelism (split-EP)** — each expert is computed on the one GPU that caches it; replication is optional caching, never a correctness requirement.
- **Superchunk prefill** *(ours)* — one expert-union fetch stream per layer per superchunk instead of per-token fetching, which turns the prefill wall length-independent (measured 8.3× fewer expert bytes, 3.9× wall); the superchunk stride is derived per box by autoconfig (P-31).
- **Two-zone VRAM expert cache** *(ours, in progress)* — a stable zone for proven-hot experts plus a fast-turnover streaming zone that lends its memory to bursty prefill/KV demand.
- **I8 placement solver** *(ours)* — a greedy + dynamic-programming hybrid that assigns work across devices of different specs (5090s next to 5080s) at the granularity of a single expert evaluation, calibrated to the box and trainable against real decode traces.
- **Speculative decoding** *(DSpark-like)* — draft-model speculation with strictly lossless batched verify and KV rewind.
- **Prefix caching** — served prompts fork from cached prefixes (measured 26 s → 3.2 s prefill on a hit), with chain-aware eviction.
- **DMA waterline queues** — per-GPU transfer queues with bounded in-flight DMA and priority staging, keeping every PCIe link saturated without flooding any single GPU.
- **Fast Python / C++ IPC** — lock-free shared-memory command/completion rings between the Python orchestrator and the C++ daemon (Cython fast path, GIL-released waits, pinned IPC region for true-async readbacks); measured orchestration residue is ~0.035 ms per decode round.
- **Persistent RAM loading** — the pinned expert store lives in a holder process and survives engine restarts: warm boots re-attach in ~83–108 s instead of rebuilding ~494 GB.
- **Expert placement statistics** — demand-fetch frequency tables (trace-fit, retrainable) drive host-arena placement, refined online by a placement migrator during serving.
- **Custom SM120 kernel optimizations** — attention, dequant, and MoE kernels tuned for consumer Blackwell (RTX 5090/5080), including split-KV decode, fused gating, and MXFP4/GGUF-native expert paths.
- **Hardware-fit config derivation (autoconfig)** *(ours)* — a CPU-only solver that derives a full, explained serving recipe from the weights + detected hardware, with hard pins, preference/accuracy levers, and refusal-over-OOM semantics ([AUTOCONFIG_MODEL.md](AUTOCONFIG_MODEL.md)).
- **Guided decoding + OpenAI-compatible serving** — xgrammar-constrained JSON/grammar output, tool-call and reasoning parsers, streaming SSE (HTTP layer overhead measured at ~0.1–0.2% of a request).

## The I8 placement model *(ours)*

Every MoE layer, the router picks `N` experts. Some are already on a GPU,
the rest are in host RAM on some NUMA bank. Which GPU should each one land
on?

The devices aren't identical (5090s next to 5080s), the banks aren't
equidistant, and the answer changes every token. So it's a solver. It
minimizes this over the assignment `j[·]`:

```
T(j) =  Σ subprep(i)                            [NVMe -> RAM staging]
     +  max( makespan , egress )                [the real bottleneck]
     +  max recon_overhead[j] + Σ recon_added[j]  [TP collective]
     +  Σ place_cons[i, j[i]]                   [may be NEGATIVE]
     +  Σ  Σ  evict_cons[j, u]                  [convex in n_j]

where

  makespan = max over devices j of
               Σ subxfer(i) + a_j·c_j + b_j·ceil(c_j / P_j)
               i on j

  egress   = max over banks b of
               ( Σ egress(i) ) · ( c_b + (1 - c_b) / g_b )
                 i from b, uncached
```

Three things make it more than a sum of latencies:

- **That `max` is two different resources.** Transfers contend on
  per-device PCIe ingest *and* on per-bank memory channels. A bank's
  channel is drawn by every fetch out of it no matter which GPU it targets
  — so the floor is the busiest bank, not the sum. You take the larger,
  never both.

- **Compute has a batch step, not a slope.** `a_j·c + b_j·ceil(c/P_j)` —
  filling a device's batch is nearly free, the `(P_j+1)`-th expert costs a
  whole new `b_j`. A linear `c·const` has no such structure and
  systematically over-spreads.

- **One term can be negative.** `place_cons` *rewards* putting a hot expert
  where it'll be reused next token. Meanwhile evictions are convex — the
  `u`-th eviction on a device costs more than the `(u−1)`-th. Concentrate
  to fill a batch, spread to dodge evictions and share links. The optimum
  is wherever those balance, and it moves with your hardware.

`c_b` is the measured contention factor of each bank: at 1 the channel is
strictly serial and the term collapses to the plain sum; at 0 it's fully
parallel and spreading across `g_b` devices divides the floor by `g_b`.
Calibration sets it per box, so the same code is inert on serial hardware
and rewards spreading on parallel channels.

Solved exactly where that's affordable — full enumeration under 2²²
candidates, subset-partition DP at `N ≤ 5` — and LPT greedy beyond, always
deterministic.

Full derivation, term by term:
**[I8_PLACEMENT_MODEL.md](I8_PLACEMENT_MODEL.md)**.
