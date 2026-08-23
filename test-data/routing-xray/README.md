# Routing x-ray artifacts (TD-PREFILL-ARENA-THRASH measured addendum)

Per-token MoE routing capture of a GLM-5.2 superchunk prefill, used to measure
expert-coverage saturation, hot-set stability, gate-threshold pruning, and the
union-growth curve U(N). Results are recorded in the TD-PREFILL-ARENA-THRASH
MEASURED ADDENDUM in `spec/TECH_DEBT.md` (2026-07-10/11).

## Files

- `route_dump_10k_k32_2026-07-10.bin` (1.6 GB, gitignored + stignored — local
  artifact, regenerate if lost; see below)
- `route_dump_10k_k32_2026-07-10.run.log` — the producing run's output
  (per-superchunk progress + `[glm52-prefill-perf]` summary: 375 layer-calls,
  mean 252.6 unique experts/layer-call, 2617 GB expert H2D, 1490 s wall)
- `analyze_route_dump.py <dump>` — (A) Jaccard(top-50 by load) hot-set
  stability across consecutive superchunks; (B) fetched-expert count per
  2048-token superchunk after excluding assignments with per-token-normalized
  gate weight < τ ∈ {0.02, 0.05, 0.08, 0.10}
- `analyze_union_growth.py <dump>` — U(N) distinct-experts-per-N-token-window
  curve vs the uniform-routing model, and residency-adjusted fetch counts with
  the previous superchunk's top-K held resident

## Dump format

One record per gating call (= one 64-token attention sub-chunk, per layer, per
TP rank — replicated ranks carry identical routing; analyzers keep the first
rank only). Written by the `LS_DRIFT_DUMP` hook (the attention-side gating
producer in `src/daemon/dispatch_attention.cpp`; record layout documented at
`drift_dump_routing` in `src/daemon/dispatch_moe.cpp`). Little-endian:

```
int32 hdr[6] = {seq, layer_idx, gpu, num_tokens, n_experts, topk}
float logits[num_tokens * n_experts]   // pre-argmax router logits
int32 idx   [num_tokens * topk]        // selected expert ids
float w     [num_tokens * topk]        // gating weights (renormalized top-8
                                       // × routed_scaling_factor; analyzers
                                       // re-normalize per token)
```

Superchunk membership is reconstructed per layer from cumulative token offset
(offset / 2048), since records arrive in sequence order.

## Regeneration

Prompt: `test-data/prompts/glm52_needle10k_tokens.txt` (10,000 ids, tracked in
git). Run (~30 min prefill + ~5 min init, GPUs 2,3):

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
GLM52_LONGCTX_TOKENS=test-data/prompts/glm52_needle10k_tokens.txt GLM52_LONGCTX_GOLDEN=101474 \
GLM52_LONGCTX_PREFILL=64 GLM52_SUPERCHUNK_K=32 GLM52_SPARSE_PREFILL=1 \
GLM52_LONGCTX_PREFILL_LIMIT=5 \
LS_DRIFT_DUMP=test-data/routing-xray/route_dump_10k_k32_<date>.bin \
./build/tests/integration/glm52_longctx_golden_test --gtest_filter='*IntGolden*'
```

`GLM52_LONGCTX_PREFILL_LIMIT=5` stops after 5 superchunks (4 full 2048-token +
one 1807-token tail) and — IMPORTANT — skips the final decode step. The
MoE-side copy of the dump hook (`dispatch_moe.cpp`, fires when gating is NOT
precomputed, i.e. decode) opens the same path with `fopen("wb")` and would
truncate the prefill capture. Remove any existing dump file before running.

## Headline results (this capture)

- Coverage saturated: 252.7/256 experts per (layer, 2048-token superchunk).
- Gate-threshold pruning dead for prefill: τ=0.05 → 252.6 fetched (0.0% saved);
  τ=0.10 → 248.4 (1.7% saved, 18.6% gate mass dropped).
- Hot-set (top-50) adjacent-superchunk Jaccard 0.43 → 0.60 → 0.74 → 0.79
  (random baseline 0.11); long-range sc0→sc4 = 0.28.
- U(N): 45.6@8, 115.6@32, 160.3@64, 226.8@256 distinct experts (uniform model:
  57.4/163.3/222.4/255.9). With prev-superchunk top-152 resident: 9.8@8,
  101.3@2048.
- Caveat: the needle10k prompt is self-repetitive filler; adjacent-J is likely
  inflated vs diverse content.
