# LayerStoRm — Architecture and Design

LayerStoRm is an inference engine for Mixture-of-Experts (MoE) language models whose weights do not fit in GPU memory. It targets medium-VRAM machines — one to four consumer GPUs of the 16–32 GB class with a large host RAM pool — and runs models such as DeepSeek V3.2 (~685 B parameters, 256 routed experts per layer) at interactive decode speeds by streaming expert weights over PCIe just ahead of need. The engine is a C++20/CUDA kernel daemon paired with a single-threaded Python orchestrator in the same process.

This document is the canonical description of the architecture. It is written so that each core mechanism can be reimplemented from the text alone; where the mechanism has a cost model, an algorithm, or a memory layout, the concrete form is given. Measured figures come from the repository's own benchmarks and are labeled with how they were obtained; analytical projections are labeled as projections.

---

## 1. Overview

### 1.1 The problem

A modern MoE model is mostly experts. DeepSeek V3.2 has 61 transformer layers, 58 of them MoE layers with 256 routed experts each; at NVFP4 quantization (0.5625 bytes/element) the routed-expert set alone is ~343 GB, while everything else the model needs per token — attention projections, gating networks, dense FFNs, shared experts, embeddings, the output head — fits in ~13.6 GB per GPU. Per token, however, only 8 of 256 experts per layer are active (~11.5 GB of expert weights touched per token across all layers — 3% of the set). The weights are enormous; the *working set per token* is small and known only after each layer's gating runs.

Naive offloading fails on this workload in specific, measurable ways:

- **mmap-backed demand paging** turns each cold 24.8 MB expert read into thousands of random 4 KB page faults: ~150 ms per expert versus 5–12 ms for a single sequential `pread` (measured; the difference alone moved the engine from 0.71 to 3.95 tok/s).
- **Synchronous whole-layer swapping** (move layer weights in, compute, move out) serializes PCIe and compute and transfers 32× more expert bytes than the token actually uses.
- **CPU-side expert compute** avoids the transfer but caps expert FFN throughput at CPU GEMM speed, and leaves GPU compute idle.

### 1.2 The core bet

LayerStoRm's bet is that on PCIe 5.0-class hosts, *moving the activated experts to the GPU* beats computing them on the CPU — provided the transfers are (a) predicted ahead of the layer that needs them, (b) fully overlapped with compute, (c) sourced from NUMA-correct pinned host memory at full link bandwidth, and (d) placed across GPUs by an explicit cost model rather than a fixed hash. Every subsystem exists to serve one of those four conditions, and a single-threaded orchestrator is the sole decision authority so that prediction, transfer, caching, and eviction can never race one another.

### 1.3 System diagram

```
 PYTHON ORCHESTRATOR (main thread — decisions only, no GPU calls)
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Predictors (signals only):  PreScope · PROBE · MoE-SpeQ · SP-MoE     │
 │        └──> Prefetch Fuser ──> TransferScheduler ──> plan             │
 │  Scheduler · EvictionPolicy · Statistics (EWMA, co-activation)        │
 │  6-phase loop: COLLECT→SCORE→STATS→PLAN→DISPATCH→YIELD                │
 └───────────────┬───────────────────────────────▲───────────────────────┘
        command SPSC ring                completion SPSC ring
        (Py→C++, 256 B slots)            (C++→Py) + seqlock StateSnapshot
 ┌───────────────▼───────────────────────────────┴───────────────────────┐
 │  C++ DAEMON (std::thread, never takes the GIL — the ONLY component    │
 │  that issues GPU commands, and only in service of ring commands)      │
 │   CommandDispatcher ─ FETCH_AND_RUN_MOE state machine                 │
 │        ├─ Expert PLACEMENT SOLVER (per token×layer, ~2.5 µs)          │
 │        ├─ ExpertLifecycleManager (NVMe→RAM→VRAM state machine)        │
 │        └─ TransferEngine (per-GPU H2D/D2H streams, dedup, waterline)  │
 └───────┬───────────────────────┬───────────────────────┬───────────────┘
         │                       │                       │
 ┌───────▼────────┐   ┌──────────▼─────────┐   ┌─────────▼─────────┐
 │ GPU 0 (TP/EP)  │   │ GPU 1 (TP/EP)      │   │ GPU 2..3 (EP)     │
 │ 7 CUDA streams │   │ 7 CUDA streams     │   │ expert cache only │
 │ pinned weights │   │ pinned weights     │   └───────────────────┘
 │ 2-zone expert  │   │ 2-zone expert      │
 │ cache + KV     │   │ cache + KV         │
 └───────▲────────┘   └──────────▲─────────┘
         │  H2D over PCIe 5.0 x16 (measured ~56 GB/s NUMA-local)
 ┌───────┴──────────────────────┴──────────────────────────────────────┐
 │ PINNED EXPERT ARENA — one slab per GPU-attached NUMA node,           │
 │ anonymous + mbind + cudaHostRegisterPortable, prepacked expert slots │
 └──────────────────────────────▲───────────────────────────────────────┘
                                │ io_uring bulk preload / direct pread
                     ┌──────────┴──────────┐
                     │ NVMe (per-expert     │
                     │ prepacked files)     │
                     └─────────────────────┘
```

Only the C++ daemon touches CUDA, and it does so only while executing commands the orchestrator placed on the ring (plus the bounded-autonomy interior of `FETCH_AND_RUN_MOE`, §6.4). The predictors compute scores; they cannot start a transfer or launch a kernel.

---

## 2. Design principles

**Single decision authority.** The orchestrator loop is the only component that decides what happens on a GPU. Prediction and prefetch modules (PreScope, PROBE, MoE-SpeQ, SP-MoE) produce `PrefetchHint` records — expert id, layer, confidence, source — and nothing else. The rationale is not stylistic:

- *Races.* A prefetcher that starts its own H2D while the eviction policy frees the same slot corrupts memory. With one decision thread, "is this slot free" has exactly one answer at any point in the cycle.
- *Ordering.* Correct MoE execution needs transfers ordered relative to compute (an expert must be resident before its GEMM). Ordering is enforceable only if a single agent sequences both.
- *VRAM accounting integrity.* The per-GPU residency map (§3.1) is the single source of truth. It stays truthful because every mutation happens synchronously inside the decision loop; there is no reconciliation step, and therefore no window in which the accounting lies.

**Two threads, zero shared locks.** The Python orchestrator (decisions) and the C++ daemon (execution) communicate exclusively through two heap-allocated lock-free SPSC rings (commands Python→C++, completions C++→Python) and a seqlock-protected state snapshot read zero-copy through numpy. One pybind11 call (`start_engine`) at startup returns the pointers; after that no Python↔C++ function call occurs on the hot path and the daemon never acquires the GIL. The daemon polls in a <20 µs cycle; the orchestrator budget is <100 µs per cycle.

**Signals cross the boundary as data, never control.** The daemon publishes state (residency bitmaps, expert statistics, cycle counters) under fine-grained seqlock transactions and yields periodically (`sched_yield` every 16 transactions) so the Python reader always finds even seqlock windows. Routing information rides in a sideband region, never inside completion structs.

**Conflicts between subsystems are resolved by explicit contracts.** The spec's conflict analysis (spec/IMPLEMENTATION_GUIDE.md §0) governs interactions such as: speculation depth is capped by current expert residency (never speculate deeper than the verification can serve); every subsystem must tolerate variable active-expert counts (adaptive top-K); expert duplication across GPUs is optional caching and never duplicates compute (§6.5); layer-skipping backs off before expert reduction when quality calibration demands it.

**No raw CUDA outside designated translation units.** All allocation, copies, streams and events go through the `DeviceBackend`/`AttentionDevice`/`ExpertDevice` interfaces; only concrete backend `.cpp`/`.cu` files may include CUDA headers. A build target (`layerstorm_no_cuda_check`) enforces this, which keeps every scheduling and policy component unit-testable on CPU — including the placement solver and the eviction policy.

**Numerics are never a knob.** Placement, eviction, prefetch and caching change *when* and *where* bytes move, never *what* is computed. The engine's EP combine is a canonical deterministic per-slot sum with unconditional cross-GPU dedup, which makes expert placement numerically inert: any placement policy reproduces the same routing trajectory bit-for-bit (validated 0/5800 layer mismatches on the reference benchmark). Consequences of this invariant appear throughout §7.

---

## 3. Memory architecture

Each GPU's VRAM is carved at initialization into fixed regions, in address order:

```
| pinned weights | KV speculation pool | indexer-K (DSA) | KV main (+ prefill
  scratch tail)  | expert STREAMING zone (spill above prefetch) | expert STABLE zone |
```

plus a safety margin left unallocated for the CUDA context and library workspaces (default 512 MB; the reference benchmark requires 2.25 GB because cuBLAS and decode transients grow with the workload).

### 3.1 The two-zone expert cache

The expert cache is a slot sub-allocator over the two expert regions. All slots are the same size — one full expert (`gate`+`up`+`down` projections at the model's quantization; 24.8 MB for V3.2 NVFP4) — so there is no internal fragmentation and free lists are plain LIFO stacks. Address = `zone_base + slot_idx × slot_bytes`.

**Stable zone.** Long-lived placements: high-frequency experts, co-activation-graph placements. Evicted only by explicit policy decision, and protected while referenced (`lock_count`).

**Streaming zone.** Fast-turnover placements: speculative prefetch, one-off demand loads. Subdivided into a *prefetch* sub-zone (bottom, permanent) and a *spill* sub-zone (top). It is the **streaming** zone — not the stable zone — that shares memory with the KV cache: the region order `kv_main → expert_streaming → expert_stable` places the spill sub-zone contiguous with the KV-main tail, and bursty KV/prefill scratch demand temporarily annexes it (`kSpillActive` mode: spill residents are evacuated by bookkeeping, spill allocations freeze, the prefetch sub-zone and stable zone continue untouched). The choice of zone is deliberate: spill residents are the cache's lowest-value, evict-first entries (streaming +500 eviction bonus, rarely lock-protected), so lending their memory to a prefill burst costs a few cheap re-fetches — annexing stable-zone memory instead would evict exactly the high-value placements the cache exists to protect, and stable entries are the ones most often pinned by `lock_count` mid-execution.

**Admission** is the two-phase `reserve` / `mark_ready` protocol:

1. `reserve(key, gpu, zone, is_duplicate)` allocates a slot and returns its VRAM address, with `sub_components_ready = 0`. Returns null if the zone is full — the caller must first plan an eviction; the allocator never evicts on its own.
2. After the DMA completes, `mark_ready(key, gpu, SubComponent)` sets readiness bits. `SubComponent{kGate=0x01, kUp=0x02, kDown=0x04, kAll=0x07}` allows partial readiness; the normal path sets all bits at once (whole-expert transfers — see §6.4 for why the engine deliberately does *not* pipeline sub-components).

Between the phases, the entry exists in the per-GPU residency map — `unordered_map<ExpertKey, CacheEntry>` — which is the **single source of truth**: kernels only ever read addresses currently in this map, and the orchestrator only mutates it synchronously in its loop.

**Eviction scoring** is stateless and policy-pluggable. The default `impact_weighted_lru` score for a candidate is

```
score = α·recency − β·frequency − γ·routing_weight − δ·temporal_autocorr
        − ε·coactivation − ζ·prefetch_score − η·hysteresis
```

with all terms normalized to [0,1] by the caller (defaults α=0.4, β=0.35, γ=0.25, δ=ε=ζ=η=0 — the extra terms are wired but off until their statistics feeds are enabled). Higher score = better victim. Two additive constants sit far above the formula's ~2.0 range and therefore act as strict priority classes: **+1000 for duplicates** (a non-primary copy always evicts before any primary) and **+500 for streaming-zone residents** (streaming turns over before stable when both are eligible). `lru` and `lfu` variants exist for comparison. Eviction victim ranking on the fetch hot path proved decisive in practice: an unranked hash-order fallback cost the cache 0.65 → 0.26 hit rate on the same routing trace versus recency-ordered selection (measured in the deterministic offline simulator, §7.6).

### 3.2 KV cache

KV memory is page-based (`PageAllocator`, 16 tokens/page default), with two pools per GPU — *main* for committed tokens, *speculation* for draft tokens — plus a separate DSA indexer-K pool. A page holds one layer's KV for a token range; for the MLA FP8 layout that is 644 B/token/layer (512 B FP8 compressed KV + 4 B scale + 128 B BF16 RoPE). Pages carry refcounts for copy-on-write forks (speculative branches share prompt pages; `cow_copy` materializes on divergence), and *promotion* of accepted speculative tokens is metadata-only: the page's pool tag flips from speculation to main, no bytes move. Headroom reservation keeps a configured number of pages available for CoW and growth so that page allocation on the accept path cannot fail mid-token. Under decode-context-parallelism, tokens are round-robin distributed across the TP GPUs in fixed chunks, so each GPU stores a disjoint 1/tp of the sequence.

### 3.3 Accounting model and invariants

- The expert cache owns VRAM expert slots; `NvmeTier` owns host RAM and NVMe. Ownership never overlaps (each byte has exactly one accountant).
- Expert data always moves NVMe → host RAM → VRAM; there is no direct NVMe→VRAM path, and the hierarchy is inclusive (a promoted copy leaves the lower-tier copy in place, so eviction from VRAM is a metadata drop, not a write-back — D2H writes exist only for genuinely dirty data).
- Monotonic progress: if VRAM is full and a demand transfer is needed, an eviction must always be possible; pinned components are excluded from expert regions precisely to guarantee this.
- A hard platform constraint discovered by measurement: GPU page tables for registered host memory cost ~3 MiB of VRAM per GiB pinned, per GPU. With ~380 GB pinned host arena this consumes >1 GB of VRAM per GPU and caps the practical pinned-fraction at 0.76 of host RAM on the reference machine (at 0.79, cuBLAS handle creation fails; at 0.85, kernels fail at launch).

### 3.4 Worked example (RTX 5090, 32 GB, DeepSeek V3.2 NVFP4, EP=2)

Measured allocation on the reference benchmark ("full-fit" configuration — the entire 343 GB routed set preloaded to host RAM):

| Region | Size | Detail |
|---|---|---|
| Safety margin | 2.25 GB | CUDA context, cuBLAS workspaces, decode transients |
| Pinned weights | 13,586 MB | attention + gating + dense FFN + shared expert + embed + head, TP-sharded |
| Expert stable zone | 10,886 MB | 460 slots × 23.6 MiB |
| Expert streaming zone | 4,665 MB | 197 slots |
| KV main + speculation + indexer | ~2.3 GB | 644 B/token/layer, 16-token pages |
| Free after init | ~0.45 GB | page-table cost of the pinned arena already deducted |

The 460+197 slots per GPU against a per-token demand of ~464 expert-layer visits is why the measured cache hit rate sits at ~0.45–0.67 depending on eviction policy quality: the cache is *capacity-bound*, and every eviction-ranking improvement translates directly into hit rate and therefore tok/s.

---

## 4. The expert streaming pipeline

### 4.1 Host-side expert pool

Experts live on disk as **prepacked per-expert files**: the exact byte image the GPU kernel consumes — NVFP4 payload with group scales pre-interleaved in the tensor-core scale-atom layout (128 rows × 4 groups), all three projections concatenated, padded to a 4096-byte slot stride. Prepacking happens once, offline (or on first run), streaming one expert at a time with atomic temp+rename writes. The payoff: an H2D transfer is a single contiguous copy with zero CPU-side packing or reformatting on the hot path.

Host RAM holds the **pinned expert arena**: one anonymous slab per *GPU-attached NUMA node* (not per GPU — a node backing two GPUs gets one shared, deduplicated arena). Each slab is `mbind`-bound to its node, prefaulted by parallel NUMA-local memset threads (~48–51 GB/s aggregate, ~8 s for 382 GB), and registered with `cudaHostRegisterPortable`. Three platform findings shape this design, all established by the repository's `bench/h2d` microbenchmark and preload instrumentation:

- Registered (not `cudaHostAlloc`'d) memory is not counted against `RLIMIT_MEMLOCK`, so hundreds of GB can be pinned without privilege changes (240 GB pinned under a 63 GB limit, measured).
- Registration costs ~50 ms/GB on prefaulted pages but ~240 ms/GB when faulting live — hence prefault first, and overlap registration (side thread) with the NVMe preload (io_uring), since they touch disjoint hardware.
- Full H2D bandwidth (~56 GB/s on PCIe 5.0 x16, measured) requires *both* a pinned source *and* NUMA locality; a cross-node source loses ~38%. Every D2H/H2D decision is therefore NUMA-aware: an expert's host copy has a *home node* chosen round-robin over GPU-attached nodes, and the consumer-side lookup prefers the arena local to the destination GPU.

Cold loads (expert not yet in the arena) use direct `pread` — O_DIRECT or buffered with `fadvise` — into the pre-pinned slot: 5–12 ms per expert versus ~150 ms for mmap demand-faulting the same bytes. Bulk preload uses io_uring (per-drive queue depth 64) and reaches the measured NVMe ceiling. On the reference machine that ceiling is 3.3 GB/s — a PCIe Gen3 x4 root-port limit, verified flat across queue depths 8–256 and block sizes 256 KB–24 MB — so full-set preload takes ~100–165 s (measured runs at 3.37 and 2.10 GB/s against the ceiling); total cold start ≈ 2.7–4 min.

The arena is slab-managed with fixed expert slots, LRU replacement, and a per-slot multi-GPU in-flight refcount: a slot is evictable only when no GPU currently sources a DMA from it and no async load is filling it. Host-source resolution for any expert walks a priority chain: pinned arena → mmap'd prepacked file → NVMe-tier host cache → retained packed buffers → lazy-pack from the original safetensors (last resort).

### 4.2 CUDA stream topology

Seven fixed streams per GPU:

| # | Stream | Role |
|---|---|---|
| 0 | `kAttention` | attention compute |
| 1 | `kExpertFfn` | expert FFN compute (grouped GEMM, SwiGLU, combine) |
| 2 | `kGating` | router/gating compute |
| 3 | `kH2dTransfer` | expert weight uploads |
| 4 | `kD2hTransfer` | evictions / exports |
| 5 | `kPrefetchCompute` | PreScope gating lookahead, PROBE predictors |
| 6 | `kAsyncDequant` | predictive KV FP8→BF16 dequantization |

Cross-stream ordering uses CUDA events only — never `cudaStreamSynchronize` on the hot path. The canonical per-layer chain: attention-complete event triggers gating (2) and PreScope (5); transfer-complete events gate expert FFN (1); FFN-complete gates the next layer's attention (0). Completion detection is non-blocking event polling in the daemon cycle.

### 4.3 Transfer engine

The `TransferEngine` owns the H2D/D2H streams and provides: **deduplication** (an in-flight key `{expert, gpu, direction}` suppresses duplicate requests, returning the existing monotonically-increasing `TransferToken`), a **priority-staged queue** per GPU with a bounded in-flight *waterline* (`max_inflight_per_gpu`), **pinned staging** for the rare unpinned source, deferred-cancel semantics (a cancelled in-flight DMA completes silently; its slot and callback are dropped only after the event fires, preventing use-after-free), and `poll_completions()` which fires callbacks and emits completion records. Demand fetches carry maximum priority and jump to the front of the staged queue ahead of any speculative prefetch.

### 4.4 Transfer scheduling

`TransferScheduler` is a stateless planner the orchestrator calls each PLAN phase:

```
plan_transfers(prefetch_priorities, resident, inflight, latency_estimates):
  candidates = [p for p in priorities
                if p.key not resident on p.gpu and not inflight]
  sort candidates by priority DESC
  for c in candidates:                      # greedy, bandwidth-constrained
    budget[gpu] = pcie_bw_gbps[gpu] * cycle_budget_us * 1e3   # bytes/cycle
    if expert_bytes > budget[c.gpu]: skip
    budget[c.gpu] -= expert_bytes
    delay = max(0, c.time_until_needed_us − transfer_latency_us[c.gpu])  # JIT
    zone  = STABLE if c.priority ≥ 0.7 else STREAMING
    emit TransferPlanEntry(c.key, c.gpu, zone, c.priority, delay)
```

The JIT delay keeps far-future predictions from occupying slots and bandwidth that near-term demand needs; the 0.7 threshold routes only high-confidence predictions into the slow-turnover stable zone. `plan_evictions` runs per GPU only when free slots are insufficient, delegating victim selection to the eviction policy with the streaming-first priority of §3.1.

### 4.5 Miss path (the synchronous fallback)

When a routed expert is not resident at MoE time, the layer's `FETCH_AND_RUN_MOE` state machine (§6.4) issues an immediate demand fetch. Cost model of the miss, measured:

- expert in pinned arena (warm): ~5–9 µs enqueue + ~0.56 ms DMA at NUMA-local bandwidth (24.8 MB / 44.5 GB/s under concurrent load), detected within the daemon's polling lag (~0.1 ms);
- expert in host RAM but unpinned: + one staging memcpy;
- expert only on NVMe (cold): 5–12 ms direct pread into an arena slot (asynchronous via the arena loader — the daemon is never blocked; the layer's compute proceeds on the resident subset and re-runs when the straggler arrives), then the DMA above.

A CPU expert device (NUMA-aware NVFP4/GGUF GEMM, ~3.2 ms per expert-layer at batch 1 on the reference host) exists as an alternative executor, but the primary architecture computes experts on the GPU where they are cached.

---

## 5. PreScope gating lookahead

**PreScope** is the project's name for exact one-layer-ahead gating: because layer *L+1*'s router input is layer *L*'s output hidden state, the orchestrator can run layer *L+1*'s gating network *during* layer *L*'s remaining work and know the next layer's expert set before that layer begins.

**Mechanism.** After layer *L*'s attention completes, the orchestrator dispatches a gating kernel for layer *L+1* on Stream 5 (`kPrefetchCompute`) against the current hidden state. The result — top-K expert indices and routing weights — returns via a completion and a pre-registered output buffer. PreScope converts it into `PrefetchHint{expert, layer=L+1, weight, confidence=HIGH, source=PRESCOPE}` records. The signal is *exact* (it is the real router), the horizon is exactly one layer, and the cost is one small extra kernel launch per layer on an otherwise idle stream.

**Learned extension.** An optional per-layer predictor (LLaPor-style) extends the horizon: a small MLP per layer group (input/middle/output thirds of the depth) maps a 256-dim random projection of the hidden state to expert-activation probabilities, runs on the CPU in ~5–50 µs, is online-trained (SGD, focal loss plus an expert-balance term) with double-buffered weight swaps, and emits MEDIUM-confidence hints. It trades exactness for horizon and zero GPU cost.

**Batch > 1.** Per token, top-K indices above a score threshold (default top-8, threshold 0.01) are extracted; across the batch the per-expert score is the *max* over tokens (an expert needed by any token must be resident), then the ranked union is emitted. This makes prediction volume grow sublinearly with batch size while remaining sound.

**Consumption.** PreScope is one of four fused prediction sources. The **prefetch fuser** combines them with time-aware decay:

```
priority(expert, layer) =
  (α·prescope + β·probe + γ·speq + δ·sp_moe) / (1 + layers_ahead · time_decay)
```

clamped to [0,1]. PROBE contributes multi-layer MLP predictions from fixed probe depths (20/50/80%); MoE-SpeQ contributes learned per-layer-pair routing predictions (CPU); SP-MoE contributes draft-token gating from the speculation subsystem (a horizon of whole *tokens*, not just layers). The fused priorities feed `plan_transfers` (§4.4). None of the predictors can start a transfer — the fuser's output is a scored wish list, and the PLAN phase decides what actually moves. Layer-skip decisions filter hints for skipped layers before they consume bandwidth.

The accuracy/latency structure is deliberate: PreScope is always right but sees one layer; the learned predictors see further but are probabilistic; SP-MoE sees across tokens but only during speculation. The fuser's weights (α…δ) and the time-decay are the tuning surface, and the JIT delay in the scheduler absorbs prediction-lead variance.

---

## 6. The orchestrator

### 6.1 The Python loop

Six phases per cycle, total budget <100 µs, no blocking calls anywhere:

```python
def run_one_cycle(self):
    # 1 COLLECT — drain completion ring (≤64/cycle): transfer done,
    #   compute done, checkpoints, expert-ready; drain new requests.
    self._phase_collect()
    # 2 PREFETCH SCORING — run PreScope/PROBE/MoE-SpeQ processing on
    #   fresh signals; fuse into PrefetchPriority list.
    priorities = self._phase_prefetch_scoring()
    # 3 STATISTICS — read seqlock snapshot: expert frequencies, EWMA
    #   workload-shift detector, co-activation graph; calibrate online.
    self._phase_statistics_read()
    # 4 PLAN — set speculation depth ≤ what residency can verify;
    #   score work items; plan_transfers(priorities, ...);
    #   plan_evictions(deficits, ...).
    plan = self._phase_plan(priorities)
    # 5 DISPATCH — write fused commands + sideband descriptors to the
    #   command ring: RUN_ATTENTION / FETCH_AND_RUN_MOE per layer,
    #   PREFETCH_BATCH, EVICT_BATCH, PRESCOPE_GATING, seq lifecycle.
    self._phase_dispatch(plan)
    # 6 YIELD — record CycleMetrics; adaptive idle wait if nothing pending.
    self._phase_yield()
```

The daemon side is a five-step polling loop (<20 µs): check shutdown flag → drain ≤64 commands and dispatch (skipping dispatch when in-flight compute ≥ its cap — the backpressure valve; commands wait in the ring, which is the queue) → poll transfer completions into the completion ring → publish state under fine-grained seqlock transactions → adaptive spin/yield.

### 6.2 Priority rules and backpressure

Work classes in descending priority: demand fetches (max priority, front of the transfer queue) > verification-critical transfers > high-confidence prefetch (stable zone) > speculative prefetch (streaming zone) > speculation compute (utility-scored; rejected outright when expert coverage for verification would fall below threshold). Backpressure operates at three points: the command ring itself (finite), the daemon's in-flight-compute cap (default 32), and the per-GPU transfer waterline. Eviction is planned only against measured deficits, never speculatively.

### 6.3 Interaction contracts

Every module the orchestrator touches has a one-directional contract:

| Module | In | Out | May touch GPU? |
|---|---|---|---|
| PreScope / PROBE / MoE-SpeQ / SP-MoE | hidden states, gating exports, snapshots | `PrefetchHint` | no (PreScope's gating kernel is dispatched *by the orchestrator*) |
| Prefetch fuser | hints | `PrefetchPriority` | no |
| TransferScheduler | priorities, residency, in-flight, latencies | transfer plan + JIT delays | no |
| EvictionPolicy | candidate stats | ranked victims | no |
| CommandWriter | plan | ring slots + sideband | no (writes memory) |
| Daemon dispatcher | ring commands | GPU work + completions | **yes — sole issuer** |

### 6.4 The scheduling-granularity invariant: one layer per slot

The orchestrator's schedulable unit for expert work is **one full MoE layer**: a single `FETCH_AND_RUN_MOE` command carries the layer's entire routed expert list, and the daemon runs a self-contained state machine — lock the resident experts, start H2D for every missing one (placement chosen by the solver, §7), run the grouped FFN on the resident subset immediately, re-run as arrivals land, finalize with the deterministic EP combine and residual. Transfers are *not* decomposed into independently scheduled micro-transfers, and experts are atomic transfer units (the `SubComponent` readiness bits exist but the engine always marks all at once).

The rationale is quantitative. At decode, one expert's transfer (~0.5–1.3 ms) exceeds its FFN compute by roughly three orders of magnitude (sub-µs GEMV per projection at batch 1). Splitting an expert's three projections into pipelined sub-transfers cannot create bandwidth — the total bytes and hence the total time are identical — and the compute overlapped is negligible; the project's analysis (spec/FINE_GRAINED_TRANSFER.md) rejected sub-expert pipelining on these grounds. What *does* matter is inter-expert and inter-layer overlap, and the layer-granular command gives the daemon exactly the visibility it needs for that: the full demand set up front (so the placement solver can balance the whole group across GPUs), plus freedom to start compute before the last transfer lands.

Consequences elsewhere: the transfer pipeline optimizes for whole-expert DMAs at a fixed size (slot allocators, stride-aligned prepacked slots, per-expert dedup keys), and the placement solver's cost model (§7) is defined per layer over the layer's expert group — its makespan terms would be meaningless if the schedule interleaved fractional experts from many layers.

### 6.5 Expert parallelism model

MoE execution is **split-EP, not replicated-parallel**: each routed expert is computed on exactly one GPU per invocation — the GPU that has (or receives) its weights — and per-GPU partial outputs are combined by a deterministic routed sum. When an expert is resident on more than one GPU (duplication is an optional caching feature), an unconditional lowest-rank dedup ensures single computation; this is enforced in the dispatcher, collective-free, from local residency bitsets. Attention, by contrast, is genuine tensor-parallel across the TP pair. This split is what makes placement purely a *performance* decision (§2, last principle).

---

## 7. The expert placement solver

The most consequential scheduling decision in the engine is made 58 times per token: *which GPU should each of this layer's routed experts be fetched to and computed on, and which residents should be evicted to make room?* The placement solver answers it with an explicit cost model, replacing two naive defaults.

### 7.1 The problem and the measured symptom

The naive placement is `gpu_idx = expert_idx % tp` — a fixed hash, blind to cache state and load; the naive eviction picks an arbitrary unlocked resident. The measured failure mode of the pair is the **cross-GPU straggler stagger**: with two GPUs fetching and computing in parallel, an imbalanced split (say 6 experts on one GPU, 2 on the other) makes the layer wait on the loaded GPU. On the reference benchmark this stagger was diagnosed at **~32 ms/token** — concurrency of ~1.52× where 2 GPUs should approach 2× — measured by a fetch-wall "x-ray" instrumentation that decomposes each layer's MoE wall time into fastest-GPU time plus dead time (per-GPU event timestamps around the fetch/compute phases of `FETCH_AND_RUN_MOE`).

Formally: per layer, assign each of the `N` routed experts (N ≤ 8 for top-8 models) to one of `M` devices (2–4 GPUs, possibly a CPU expert device), given `B` NUMA banks holding the host copies — a variant of unrelated-machines makespan scheduling (`R||Cmax`) with additive assignment costs and a convex per-device eviction load. The decision variable is the assignment vector `j[1..N] → {1..M}`; `G_j` is the group on device `j`, `c_j = |G_j|`.

### 7.2 The objective (spec/GPU_LOADER_MODEL.md is authoritative for this math)

```
T(j[·]) = prep + max(device_makespan, bank_egress) + recon
        + place_cons_total + evict_cons_total          — minimize over j[·]
```

**prep** — NVMe→RAM staging for any expert not yet in host RAM: `Σᵢ subprep(i)`, with `subprep = 0` if resident in RAM else the NVMe read time. Assignment-independent; a constant offset carried so the prediction is a true end-to-end time.

**device_makespan** — the destination-side roofline, grouped by GPU:

```
subxfer(i) = 0                                          if cached_gpu[i, j[i]]
           = ncf[tier(numa_bank[i], j[i])] · xfer_speed[j[i]] + xfer_lat[j[i]]

R_j = Σ_{i∈G_j} subxfer(i) + compute_j(c_j)             (barrier mode — current engine)
device_makespan = max_{j∈P} R_j
```

`ncf` is the NUMA-closeness multiplier (measured ≈1.36 for a cross-node source), `xfer_speed` the per-expert tier-1 ingest time (~557 µs at 24.8 MB / 44.5 GB/s under load), `xfer_lat` the fixed per-transfer latency (~6–9 µs enqueue + ~110 µs detection bucket). Compute is **sub-additive, not linear**:

```
compute_j(c) = a_j·c + b_j·⌈c/P_j⌉
```

a per-expert linear cost `a_j`, a fixed cost `b_j` per *batch*, and batch width `P_j` (a GPU grouped-GEMM's saturation width, or a CPU device's thread count) — fitted by calibration sweeps (measured on the reference GPU: a≈16.6 µs, b≈64.5 µs, P=64 for the full routed-FFN chain). The batch step creates a genuine concentrate-vs-spread trade: filling a batch amortizes `b_j`, but the (P+1)-th expert opens a new batch. *Barrier mode* reflects the current engine (one grouped GEMM per device gated on all its arrivals); a *pipelined* mode (two-corner roofline `max(transfers + one-expert tail, max-transfer + batched compute)`) is defined for a future per-arrival execution.

**bank_egress** — the source-side floor, grouped by NUMA *bank*, which captures two GPUs pulling from one bank:

```
raw_sum_b = Σ_{i: numa_bank[i]=b, uncached} egress(i)        # shared channel, serial sum
g_b       = #distinct devices receiving ≥1 uncached expert from bank b
bank_egress = max_b raw_sum_b · (c_b + (1−c_b)/g_b)          # contention-aware
```

`c_b ∈ [0,1]` is a calibrated per-bank contention factor: `c_b=1` (strictly serial channel — the reference machine) makes the factor 1 and the term a pure floor; `c_b=0` (parallel channel) divides the floor by the number of distinct destinations, rewarding spread. The crucial structural fact: an expert's source bank is fixed and independent of `j[i]`, so with `c_b=1` the bank term depends on the assignment only through *which* experts are fetched (the caching pattern), never *where* they land — which is why the exact solver below can remain a per-device subset partition, with the bank floor applied as `max(makespan, floor)` per candidate rather than a cross-device coupling inside the recursion.

**recon** — the TP reconciliation (routed-output allreduce) after compute: `max_{j∈P} recon_overhead[j] + Σ_{j∈P} recon_added[j]` — the slowest participant's fixed overhead plus per-participant payload contributions. Calibrated against the real cross-GPU NCCL sum-allreduce of the BF16 routed output (a tiny-payload floor plus a full-payload term). Small on the reference hardware (~4% of predicted time).

**place_cons_total** — `Σ place_cons[i, j[i]]` over *newly placed* (uncached) experts: the future-token consequence of a placement. This is the model's only signed term — a placement that future tokens will reuse earns a negative (reward) value. The statistics behind it (co-activation, recency, predicted reuse) are orchestrator-fed over all experts; the daemon extracts the N×M sub-table per solve. It is fed 0 today, and the default mode is **clamp0** — see §7.5.

**evict_cons_total** — the future cost of the evictions each placement forces: `n_j` = new experts on `j` ⇒ `n_j` evictions, and

```
evict_cons_total = Σ_j Σ_{u=1..n_j} evict_cons[j, u]
```

where `evict_cons[j,·]` is `j`'s best-victim list cost in ascending-victim order — non-decreasing in `u`, so the term is **convex in `n_j`**: piling fetches onto one device gets progressively more expensive, pushing toward balance. Delivered to the solver as a per-device prefix-sum `evict_cum[j][n]` (O(1) read per candidate). Because these are *future* milliseconds added to *this layer's* milliseconds, they carry a horizon discount γ (default 0.1); undiscounted, the eviction term was 57.6% of predicted time — a phantom that mis-ranked assignments.

An assignment is infeasible (cost +∞) if `n_j` exceeds device `j`'s evictable candidates.

### 7.3 The algorithm

One solver, parameterized by `C` = the number of experts solved exactly; preallocated flat arrays (kMaxExperts=64, kMaxDevices=16, kMaxBanks=16, kMaxC=5 for the DP residual, ~7 KiB scratch, zero heap on the hot path); integer fixed-point nanoseconds and lowest-index tie-breaks throughout, so every solve is deterministic and every TP rank recomputing it gets the identical answer.

**Tier 1 — exact DFS branch-and-bound** when `M^N ≤ 2²²` (for M=2 that is `2^N`, i.e. any N ≤ 22 — the production regime): depth-first over experts, maintaining incremental per-device sums; prune when a monotone lower bound of the partial assignment meets the incumbent. With a partially-parallel bank (`c_b<1`) the true floor is non-monotone in assignment progress, so the bound uses the optimistic `g_b = M`; leaves evaluate the exact floor. Full objective, provably optimal. Measured ~2.5 µs/solve at N=8, M=2 including the O(N·M) cost-matrix build — ~0.15 ms/token across 58 layers, 0.08% of a 190 ms token.

**Tier 2 — subset-partition dynamic programming**, `O(M·3^N)` (linear in M), when enumeration is too large but N is small:

```
dp[j][S] = min over T⊆S of  combine(dp[j−1][S\T], cost_j(T))
cost_j(T): R_j = init_sub_j + Σ_{i∈T} subxfer[i][j] + compute_j(init_cnt_j + |T|)
A_j(T)   = recon_added_j·[T≠∅] + Σ_{i∈T} place[i][j] + evict_cum[j][|T|]
```

Per device, all `(S, T⊆S)` pairs are enumerated by Gray-code submask iteration (3^N total); ping-pong DP layers; backtrack recovers `j[·]`. The DP is exact for the makespan (`max_j R_j`, with the bank floor folded per candidate); the separable consequence terms are evaluated as a post-pass/tie-break — exactness with signed `place_cons` live is what Tier 1 is for.

**Tier 3 — greedy-prefix + exact residual** past the N≈12 knee: compute each expert's solo cost `min_d(subxfer + compute_d(1))`; run three greedy routes to completion — (a) unbounded per-expert best-device, (b) the same excluding route (a)'s choices (diversification), (c) LPT: descending solo cost, each to the device minimizing the resulting bottleneck (the classic 4/3-approximation order). Each route freezes the `N−C` most *confident* decisions (largest margin between best and second-best device; route (c) freezes the costliest first) as per-device initial state (`init_sub`, `init_cnt`, frozen eviction counts), then runs the exact DP on the remaining `C` (default 5); return the best of the three residual solutions and a pure greedy. ~13–15 µs at N=8/M=10.

**Fallback — stateful LPT list-scheduling**, `O(N·M·(M+B))`: because the greedy maintains running per-device sums, it prices the *exact* marginals — `compute_j(c+1) − compute_j(c)`, the convex eviction increment, the updated rooflines — via apply/evaluate/undo per candidate; its only weakness versus the DP is lookahead. A **streaming variant** emits each `j[i]` as decided so dispatch can begin before the solve completes; with solves at ~µs and transfers at ~600 µs this buys little today and exact-then-dispatch is the default, but it is the substrate for future work-stealing (feeding a device that finishes early).

One subtlety the tiers all respect: a cached expert is an everyday per-`(i,j)` zero inside the cost function, **never** a pre-reduction. Pre-pinning cached experts to their caching devices misses the straggler-relief optimum — when the caching device is the bottleneck, paying a transfer to move a cached expert's *computation* elsewhere wins.

**Worked example** (from the CPU unit tests, which cross-check the solver against brute force). Two devices, eight uncached experts, all from one serial bank with dominant egress: the bank floor is identical for *every* split, and the makespan nearly so — only the eviction term distinguishes. Without `evict_cum` the deterministic tie-break piles all eight on device 0 (8/0). With a strictly convex victim curve `evict_cum[n] = 10·(1+2+…+n)`, the solver returns 4/4, since `2·cum[4] = 200 < cum[8] = 360`. The convex eviction curve *is* the load-balancer of last resort when the transfer terms cannot discriminate.

### 7.4 Calibration

All constants come from an on-box calibration (`src/core/gpu_loader/loader_calibration.*`): the GPU×NUMA H2D matrix over portable-pinned NUMA-bound buffers with a rotating footprint (defeats HBM caching so rates are DDR-bound), per-bank egress and same-bank two-device contention (`c_b`), the compute curve `(a, b, P)` fitted by grid-search-P + least-squares over grouped-GEMM sweeps, and reconciliation via a real NCCL allreduce of the routed-output payload. Calibration JSON is weight/config-specific and lives adjacent to the weights; the engine rejects and re-runs a calibration whose compute dimensions do not match the deployed model. One instructive correction is now an invariant: an idle PCIe link sits at gen1 under ASPM, so a calibration that times device 0 on a cold link records ~half its real bandwidth and fabricates a phantom device asymmetry (the early solver packed 61% of experts onto the "cheaper" GPU because of it); calibration therefore performs a sustained ~20 ms link warmup before timing.

### 7.5 Accuracy/speed levers

Under budget pressure the model sheds terms in a fixed order (cheapest fidelity loss first):

1. **`place_cons` mode** — `clamp0` (default): clamp to ≥0, keeping the penalty half ("don't waste a slot on an expert that is rare or already resident elsewhere") and dropping the reward half. Rationale: the signed term is the only one that breaks non-negative makespan structure — a reward can make a worse-makespan assignment win overall, which invalidates branch-and-bound monotone pruning and greedy marginal reasoning. `off` drops it entirely; `signed` is the full-accuracy research mode.
2. Drop the bank-egress floor when banks are never binding (all fetches NUMA-local).
3. Drop `recon` from the argmin when `recon_added` is assignment-invariant.
4. Replace exact with LPT.

The intent: missing or zeroed factors coarsen the *prediction*, never corrupt *correctness* — as the orchestrator supplies better statistics, routing sharpens.

### 7.6 Invariants, validation, and current standing

**Numerics-unchanged.** The solver changes only where each expert is fetched/computed and which victim is evicted; the routed-FFN math, the deterministic EP combine, and the produced logits are identical. With the engine's canonical per-slot combine and unconditional EP dedup, placement is *numerically inert* in the strongest sense: every placement/eviction policy reproduces the identical routing trajectory bit-for-bit (0/5800 layer mismatches across policies on the reference trace). Validation is therefore tok/s plus golden-token identity — no numeric thresholds change with placement.

**Evaluation methodology.** Placement/eviction policies are judged in a fixed gate order: first a deterministic CPU **offline simulator** that replays a fixed canonical routed-expert trace through the real solver + expert cache + eviction code (bit-reproducible; the primary hit-rate gate — it predicts engine tok/s within ~2% for non-forking policies), then keeper tok/s averaged over prompts. Model-fit metrics (R², predicted-T) are explicitly *not* the acceptance metric: they were observed to diverge from tok/s, and a solver that provably minimizes a myopic objective still loses end-to-end — the objective, not the solve, is what tok/s validates.

**Standing.** The solver machinery is validated (per-layer model R²≈0.74, Spearman 0.91; 5800/5800 layers provably optimal for the objective) and *acting* on `j[·]` is a net win on the reference benchmark: 7.03 tok/s vs 6.95 baseline on the canonical trace, with the fetch-wall dead time roughly halved (~46 → ~22 ms/token), at a hit rate ≈ baseline (~0.01 intrinsic reroute-locality cost). The known open lever is that the objective is single-layer: the cross-token reuse reward that `place_cons` was designed to carry is still fed 0, and filling it (co-activation/recency/predicted-reuse statistics from the orchestrator) is where placement is expected to move from "small win" to "structural win."

**Position in the architecture.** The solver is the cost core of the daemon-resident *living router* — the component that owns expert→GPU choice and victim choice at the fetch site across its three modes (guided by an orchestrator-provided map, semi-guided by scores, or unguided/local). It replaces the `e % tp` harness placement, its `evict_cum` input is the eviction-policy scores in prefix-sum form, and the Python-side placement/load-balancing modules become *statistics feeders* (`place_cons`, `evict_cons`, candidate sets), not deciders.

---

## 8. Attention backend integration

Attention is pluggable behind a single interface. `AttentionDevice` (one per TP GPU) merges the hardware primitives (GEMM, RMSNorm, FP8 quantize, alloc/copy/streams/events) with the attention pipeline (KV append, prefill, decode, distributed-context execution); `ExpertDevice` (one per GPU with an expert cache) carries the MoE side (grouped GEMM over NVFP4/FP8/GGUF, SwiGLU, permute/unpermute). The attention algorithm is baked into the concrete type at construction via a factory — there is no per-call dispatch — and TP GPUs get both device objects while expert-only GPUs get just an `ExpertDevice`.

The primary backend, `SnapMlaSm120AttentionDevice`, implements the SnapMLA recipe — FP8 compressed-latent KV with a per-token scale and BF16 RoPE (the 644 B/token/layer layout of §3.2) — through the project's own SM120 kernel collection (`deps/LayerStoRmKernels`): dense and sparse FP8 split-KV decode in absorbed mode (the query projection fused with the KV latent absorption), dense and sparse (DSA top-K) prefill, the log-sum-exp combine across splits, the decode-scheduling metadata kernel, and the preparation kernels (fused Q quantize, K append, indexed compressed-KV dequant, `q_absorb`, RoPE rotation). These kernels are adapted from DeepSeek's FlashMLA (Apache-2.0) — parameter layouts and the split-KV combine math retain parity with it — and rebuilt for SM120 (GeForce Blackwell) around the FP8-KV pipeline, which the upstream kernels do not provide. The original FlashMLA sources remain vendored under `3rd-party/FlashMLA/` (with its CUTLASS submodule) and are still compiled into the build as a reference object library, but the runtime attention path is the SM120 collection. Around the kernels the backend owns the DSA sparse-attention indexer and its dedicated KV pool, KV quantization staging, the DCP correction stitch after distributed attention, and all stream/event wiring.

What is generic vs backend-specific: everything above the `AttentionDevice` interface — the orchestrator, transfer pipeline, memory management, MoE path — is backend-agnostic and CUDA-free; everything below it (kernel choice, KV page byte layout parameters, quantization staging) is the backend's private business. A second backend, `TqSm120AttentionDevice` (TurboQuant 4-bit KV, 386 B/token/layer), exists behind the same interface, which is the practical proof of the boundary.

---

## 9. Comparison with prior systems

The mechanisms most similar to LayerStoRm's appear in systems with different bets. The comparisons below state each system's actual mechanism first; several of the research systems' papers are vendored in `ref/layerstorm-papers/` and informed this design directly.

**llama.cpp (layer/tensor offloading).** llama.cpp statically partitions weights at load time — N layers (or per-tensor overrides) to GPU, the rest computed on CPU with its highly-optimized quantized GEMM. Data does not move at inference time; compute goes where the weights are. This is robust and simple, and for pure-CPU or tiny-GPU machines it is the right design — LayerStoRm's own CPU expert kernels vendor code from the ik_llama lineage precisely because that CPU GEMM is excellent. The architectural difference: for a 343 GB expert set and a 32 GB GPU, static partitioning pins >90% of experts to the CPU permanently, so expert FFN throughput is CPU-bound forever. LayerStoRm keeps the *placement dynamic*: any expert can be on any GPU next token, so GPU compute applies to whatever the workload actually routes to, at the price of a prediction-and-transfer machine llama.cpp does not need.

**ktransformers (heterogeneous CPU/GPU MoE inference).** ktransformers also computes attention/dense parts on GPU and routed experts on CPU (AMX/AVX-optimized), i.e., it embraces compute-where-data-lives for experts and invests heavily in CPU kernel speed and CPU/GPU work partitioning. It achieves strong results, particularly in prefill, on machines with powerful many-core CPUs. LayerStoRm makes the opposite bet for decode — stream the ~8 activated experts per layer to the GPU — which wins when PCIe bandwidth × hit rate beats CPU GEMM throughput (§10) and loses when it doesn't (weak PCIe, strong CPU). LayerStoRm's multi-NUMA CPU expert device is essentially the ktransformers position held in reserve as a fallback executor.

**PowerInfer (hot/cold neuron offloading).** PowerInfer exploits *activation sparsity within FFNs* of ReLU-family dense models: a minority of "hot" neurons fire for most tokens and live on the GPU; cold neurons are computed on CPU on demand. It is neuron-granular and depends on ReLU-style sparsity statistics. LayerStoRm operates at expert granularity on MoE models, where sparsity is *structural* (the router names the active experts exactly) rather than statistical — so prediction can be exact one layer ahead (PreScope) instead of learned activation predictors, and the transfer unit is a 24.8 MB expert rather than neuron rows. For dense ReLU models on a single consumer GPU, PowerInfer's design applies where LayerStoRm's simply does not.

**MoE-Infinity (activation-aware expert offloading).** MoE-Infinity traces Expert Activation Matrices per request and prefetches/caches experts based on sequence-level activation patterns. It shares LayerStoRm's premise (experts stream from host to GPU) and differs in the decision machinery: LayerStoRm fuses an *exact* one-layer signal with learned multi-layer and cross-token predictors, runs a per-layer placement solver across multiple GPUs with an explicit roofline objective, and manages VRAM with the two-zone/slot design. MoE-Infinity's request-level tracing is a lighter-weight approach well matched to serving many short requests.

**fMoE (fine-grained expert offloading).** fMoE tracks iteration-level "expert maps" (router probability distributions) and uses semantic + trajectory similarity to guide prefetch and eviction — finer *temporal* granularity of prediction, expert-atomic transfers like LayerStoRm. LayerStoRm's prefetch fuser is philosophically similar (multiple probability signals combined), with the addition of the exact PreScope signal and the placement/eviction cost model downstream. (A note for readers of this repo's history: sub-*component* transfer pipelining is sometimes attributed to fMoE; the paper is expert-atomic, and LayerStoRm's own analysis rejected sub-expert pipelining — §6.4.)

**MoE-Lightning / DALI (roofline-driven batch offloading).** These systems drive CPU-GPU MoE pipelines with hierarchical roofline models to maximize *throughput* under memory constraints, padding batch sizes to keep every channel busy. They are the right shape for offline/batch serving. LayerStoRm's cost model is a close cousin (its makespan+bank-egress objective is a roofline), but pointed at per-token *latency* at batch ≈ 1–few, where JIT transfer timing and straggler balance dominate rather than batch shaping.

**DeepSpeed-Inference/MII and vLLM CPU-offload.** Both provide swap-based weight or KV offload at coarse granularity, designed around synchronous or layer-wise movement within throughput-oriented serving stacks; vLLM's strength is scheduling and paged KV at scale on weights that fit. Neither attempts predictive per-expert streaming with multi-GPU placement; conversely, LayerStoRm does not attempt their continuous-batching serving breadth — its serving layer is a thin HTTP wrapper over one engine.

Where prior systems are simply better: pure-CPU boxes (llama.cpp), strong-CPU/weak-PCIe machines and prefill-heavy work (ktransformers), dense ReLU models (PowerInfer), high-throughput batch serving (MoE-Lightning-style designs, vLLM). LayerStoRm's niche is *latency-interactive decode of very large MoE models on a small number of consumer GPUs with fast PCIe and abundant RAM*.

---

## 10. Performance model

### 10.1 When expert streaming wins

Per token, the expert bytes that must cross PCIe are:

```
bytes/token = L_moe × K × (1 − h) × S_e
```

with `L_moe` MoE layers, `K` activated experts/layer (dedup within a layer is negligible at K=8, E=256), `h` the cache hit rate, and `S_e` the packed expert size. For V3.2 NVFP4: 58 × 8 × (1−h) × 24.8 MB ≈ 11.5 × (1−h) GB/token. The transfer-bound decode floor across G GPUs fetching in parallel with balanced groups is `bytes/token / (G × BW_eff)`. Measured anchors on the reference machine (2× RTX 5090 on PCIe 5.0 x16, 4-NUMA-node host): `BW_eff` ≈ 44.5–56 GB/s per GPU (loaded vs. ideal, NUMA-local pinned source; −38% if cross-NUMA), per-transfer overhead ~6–9 µs + ~0.1 ms detection.

Worked check against measurement: at the measured h=0.45, bytes/token ≈ 6.3 GB — exactly what the benchmark reports — giving a two-GPU transfer floor of ~57–71 ms/token; the benchmark's fetch wall (~46 ms fastest-GPU + stagger) matches once per-transfer overheads and imperfect overlap are added. At h→1 the floor vanishes and decode approaches the compute-bound regime (attention ~1.8 ms/layer measured under concurrent H2D, MoE ~0.7 ms/layer).

Streaming beats CPU expert compute when `S_e/BW_eff + compute_gpu < compute_cpu` amortized at the achieved hit rate: per expert-layer, ~0.56 ms transfer + ~0.15 ms GPU compute versus ~3.2 ms measured CPU GEMM (batch 1, one NUMA node) — a ~4× margin per *missed* expert, and far larger per *hit*. The margin inverts on PCIe 3.0-class links (×4 lower bandwidth) or with much faster CPUs — that is ktransformers territory (§9).

### 10.2 Measured trajectory (how obtained)

All decode numbers below come from the repository's keeper benchmark — a 100-token greedy decode integration test with full routed-expert LRU caching, EP=2, DeepSeek V3.2 NVFP4, golden-token-identity checked (0 NaN, bit-identical reference continuation) — run on the reference machine. They are point measurements from that test's own timing report, not projections:

| Stage | tok/s | What changed |
|---|---|---|
| mmap demand-fault cold loads | 0.71 | baseline staging |
| direct pread cold loads | 3.95 | kill page-fault storms (§4.1) |
| fully-correct baseline after numerics campaign | 2.93 | six numerics bugs fixed; real routing diversity restored |
| progressive `FETCH_AND_RUN_MOE` | 4.21 | fetch/compute overlap + cross-GPU H2D overlap |
| + solver-blessed eviction map | 4.95 | ranked victims at the fetch site |
| full-arena warm preload + subsequent kernel/dispatch optimization | ~6.9–7.2 | io_uring preload default; measured baseline 6.945 |
| + placement solver acting (ACT) | 7.03 | §7.6; canonical-trace A/B vs 6.95 |

Setup cost (full-fit): ~8 s arena prefault + ~100–165 s io_uring preload against the 3.3 GB/s disk ceiling (overlapped with ~22 s registration) + ~35 s TP weight upload ≈ 2.7–4 min to first token, dominated by the NVMe link.

The deterministic offline simulator (§7.6) reproduces engine tok/s within ~2% for placement/eviction variants, which is what makes cache-policy iteration cheap.

### 10.3 Where the approach degrades

- **Working set ≫ VRAM slots.** The hit-rate term is the whole game; with 657 slots/GPU against 464 visits/token, routing diversity directly sets h. A workload with flat expert usage (no head experts) pins h near the capacity ratio and decode approaches the transfer floor.
- **PCIe generation.** Halving link bandwidth (gen4) roughly doubles the miss cost; at gen3 the streaming bet is likely wrong versus a strong CPU.
- **Batch size.** Larger batches activate more *unique* experts per layer, but sublinearly (toward saturation at all 256), so transferred bytes *per generated token* fall while compute per transferred byte rises — streaming amortizes *better* with batch until unique-expert coverage saturates, after which the design converges to "resident everything that matters," i.e., the problem disappears into capacity.
- **Prefill.** Long-prompt prefill activates essentially all experts every layer; the engine chunks prefill and reuses the spill zone as scratch, but prefill throughput is not this architecture's strength — batch-roofline designs (§9) do prefill-heavy work better.
- **Cross-NUMA topologies.** If GPUs concentrate on one node, the home-node partition loses locality for half the arena; the −38% penalty then applies to a fraction of transfers (mitigated, not eliminated, by consumer-aware source selection and optional cross-node spill weighting).

---

## 11. Specified designs, not yet implemented

The three designs below are **specified but absent from the current implementation** (their full specifications live in `spec/FUTURE.md` F9–F11). They are documented here at enabling depth because each completes an identified gap in the shipped system. Nothing in this section should be read as an existing feature.

### 11.1 Per-device kernel-strategy solver (F9)

The placement solver fixes *where* each expert computes; a second, downstream decision remains fixed today: *how* each device executes its group — the engine always runs one grouped GEMM per device per layer pass. The kernel-strategy solver makes that choice per device per layer among:

1. **Grouped GEMM** (current default): one batched kernel over all `c_j` experts; best amortization (`b` cost paid once per `P`-wide batch), but barrier semantics — waits for all arrivals.
2. **Sequential per-expert execution** (dequant-GEMV at decode): compute each expert the moment its transfer lands; worst amortization, best overlap — turns the device roofline from barrier mode into pipelined mode (§7.2).
3. **Parallel batched execution**: split the group into sub-batches on independent streams; intermediate amortization, partial overlap; useful when `c_j` spans multiple `P`-widths or arrival times are bimodal.

It runs *after* the placement solve, when `j[·]` and the transfer enqueue order are known — its inputs are per-device: the group size `c_j`, the token count (GEMM M-dimension), the predicted per-expert arrival times from the transfer plan, and per-strategy calibration curves. The cost model is the same family as §7.2's compute term, one curve per strategy `s`: `compute_j^s(c, m) = a_s(m)·c + b_s(m)·⌈c/P_s⌉`, fitted by the existing calibration harness swept per strategy; the chosen strategy minimizes predicted *finish* time `finish(s) = max(last_needed_arrival(s), start) + compute^s(...)`, where sequential strategies need only the *next* arrival, not the last. Because inputs are frozen at solve time, the decision is pure and deterministic like the placement solve, and it feeds back: the constants used by the *next* placement solve become strategy-conditioned, so the two solvers converge on consistent predictions. Expected first win: eliminating the barrier stall when one straggler transfer delays an otherwise-resident group — precisely the case measured in the fetch-wall x-ray.

### 11.2 Predictive H2D backfill (F10)

The transfer plan (§4.4) is budgeted per cycle, and the schedule leaves the H2D link idle in identifiable windows — attention phases, layers with high hit rate, cycles whose demand is below budget. Backfill fills those windows with *speculative, not-yet-demanded* expert transfers drawn from the prefetch fuser's tail (candidates below the normal dispatch threshold). Admission requires all of: fused priority above a floor; a predicted idle window ≥ one expert transfer time on that GPU's link; a free *streaming-zone* slot, or a victim whose eviction score exceeds a safety margin (backfill must never force a stable-zone eviction or displace a planned one); in-flight waterline occupancy below a configured share reserved for demand (complemented by the specified cancelable no-op placeholder mechanism, which parks removable no-op entries in the in-flight window so a demand fetch can always claim a slot instantly, since a real in-flight DMA is uncancelable). Every backfilled transfer is tagged; if evicted unused it is charged as waste, and the controller throttles the priority floor on a waste-ratio EWMA (target ≤ ~0.5) — waste costs only spare bandwidth by construction, but the accounting keeps the streaming zone from churning. The measured miss anatomy motivates the design: ~254 fetched experts/token at h=0.45 are demand-timed today; every one converted to a backfilled arrival removes its detection-plus-schedule latency from the layer's critical path.

### 11.3 Pinned-layer weight rotation during attention (F11)

The pinned region (§3.4: ~13.6 GB/GPU, ~223 MB per layer per GPU TP-sharded) buys zero-latency access to per-layer weights that are each needed for ~2 ms per token. Rotation reclaims part of that VRAM: keep a sliding window of `W` layers' pinned weights resident, and during layer `L`'s attention (a transfer-quiet phase — expert demand for layer `L` starts only after gating), prefetch layer `L+W`'s pinned bundle from the host arena into the rotation buffer. Sizing is the crux and the reason only a *few* layers can rotate: one layer's bundle at ~223 MB needs ~4 ms at 56 GB/s — about two attention windows (~1.8 ms each) of lead per rotated layer — and the *sustained* cost of rotating `R` layers is `R × 223 MB` per token of H2D that competes with expert streaming (each rotated layer ≈ 9 experts' worth of bandwidth). With measured link slack of ~1–2.5 GB/token on the reference workload, `R ≈ 4–10` layers is the realistic ceiling; each rotated GB returns ~40 expert slots, so the trade is favorable exactly when the hit rate is capacity-bound (it is — §3.4). Rotation transfers ride the H2D stream at a priority above backfill and below demand; a layer's bundle must be resident before its attention issues, with a synchronous demand fetch (and a stall) as the correctness fallback; the rotation set should exclude the first/last layers and any layer on the speculation verification path, whose timing is least predictable. Numerics are unchanged — the same weights compute in the same place, only their residence is time-shared.

### Speculation subsystem (implemented; noted for completeness)

Speculative decoding is implemented in the orchestrator (drafters including the model's native multi-token-prediction head, prompt-lookup, self-speculative reduced-expert/layer-skip drafts; a draft combiner; verification with CoW KV forks and metadata-only promotion, §3.2). Its architectural attach point is the draft-combiner interface, where external drafter models could also plug in; its most important interaction with this document's subject matter is SP-MoE — draft-token gating outputs feed the prefetch fuser, extending expert prediction across *token* boundaries. A specified extension of that idea hangs LoRA expert-prediction heads off the frozen MTP trunk (adapters on the last few MTP blocks): because LoRA is additive, one pass yields both streams — the token head reads bit-identical base activations, leaving acceptance rate unchanged by construction, while the heads predict the next token's per-layer expert sets *marginal over next-token uncertainty*, a strictly better prefetch target than SP-MoE's single sampled draft path.

---

## 12. Failure modes and limitations

- **Capacity-bound hit rate.** On the flagship model the VRAM expert cache covers ~1.4× one token's expert visits; hit rate (0.45–0.67 measured, policy-dependent) — not kernel speed — is the first-order determinant of decode throughput. Workloads with flat routing distributions will do worse than the benchmark.
- **The pinned-fraction cliff.** GPU page tables for registered host memory (~3 MiB VRAM per pinned GiB per GPU) create a hard, initially-invisible ceiling: exceed ~0.76 pinned fraction of a 512 GB host on this configuration and CUDA library initialization fails in obscure ways. The engine sizes around it, but it is a real deployment foot-gun on bigger hosts.
- **Cold start.** Minutes to first token at full-fit (NVMe-bound; ~3.3 GB/s on the reference machine's Gen3 x4 root port). The design assumes a long-lived process.
- **Prefill.** Chunked and correct, but not competitive with batch-roofline designs on long prompts (§10.3).
- **Placement gains are partially unrealized.** The placement solver's cross-token reuse term (`place_cons`) is unfed; today's ACT win (7.03 vs 6.95 tok/s) comes from fetch balancing alone. Until the orchestrator feeds reuse statistics, `e % tp`'s free placement *stability* remains a nontrivial fraction of what the solver has to beat.
- **Determinism scope.** Single-run determinism holds (deterministic reduce default-on; placement numerically inert). Reduction-order determinism across ≥3 NCCL ranks is not yet guaranteed — the reference configuration is 2-rank.
- **Single-node scope.** No multi-node execution; parallelism is TP/EP/DCP within one host.
- **Orchestrator budget.** The Python loop's <100 µs cycle is measured-adequate at batch ≈ 1–few with fused commands (~183 commands/token), but it is a standing constraint on how much per-cycle intelligence (bigger predictors, bigger fusion) can live on the hot path; heavier machinery belongs on the daemon side or in background threads, as the placement solver already demonstrates.
- **Online predictors cold-start.** Learned predictors (PROBE, MoE-SpeQ, learned PreScope) begin untrained per deployment; until their online training converges, prefetch quality rests on exact-PreScope alone (one layer of lead).

---

## 13. Glossary

- **ACT / shadow mode** — placement-solver operating modes: *shadow* computes and logs `j[·]` while the engine routes by `e % tp`; *ACT* routes and evicts by the solver's output.
- **Bank egress floor** — the per-NUMA-bank component of the placement objective: the serialized drain time of all uncached transfers sourced from one bank (§7.2).
- **DCP** — decode context parallelism: sequence tokens round-robin across TP GPUs with per-GPU partial attention and a correction combine.
- **DSA** — the sparse-attention indexer of DeepSeek-V3.2-class models; has its own KV pool (indexer-K).
- **EP (split-EP)** — expert parallelism in which each routed expert computes on exactly one GPU per invocation (§6.5).
- **ExpertKey** — (layer, expert) identity used across all tiers and maps.
- **Expert lifecycle manager (ELM)** — the daemon-side state machine driving an expert through NVMe → arena → VRAM with completion events per stage.
- **FETCH_AND_RUN_MOE** — the layer-granular fused command: fetch missing experts (placement-solved), run the routed FFN progressively on arrivals, finalize with the deterministic combine (§6.4).
- **Full-fit** — the benchmark configuration with the entire routed-expert set preloaded into the pinned host arena.
- **Golden test** — bit-identity check of a reference greedy continuation; the correctness gate for every performance change.
- **Home node** — the NUMA node designated to hold an expert's host copy (round-robin over GPU-attached nodes).
- **Keeper** — the standing 100-token decode benchmark used as the acceptance gate for performance work.
- **Living router** — the daemon-resident placement/eviction decision component; the placement solver is its cost core (§7.6).
- **NVFP4** — 4-bit float (E2M1) weight format with UE8M0 group scales, 0.5625 bytes/element.
- **Offline simulator** — the deterministic CPU replay of a canonical routed-expert trace through the real cache/solver/eviction code; primary gate for placement/eviction policy work.
- **Pinned expert arena** — the per-NUMA-node registered host slab holding prepacked experts (§4.1).
- **place_cons / evict_cons** — the placement objective's future-token corrective terms (§7.2).
- **Prepacked format** — the on-disk per-expert byte image with tensor-core-interleaved scales and 4096-byte slot stride, consumed by DMA without repacking.
- **PreScope** — exact one-layer-ahead gating on the prefetch stream (§5).
- **PROBE / MoE-SpeQ / SP-MoE** — learned multi-layer, learned per-layer-pair, and speculation-driven expert predictors feeding the prefetch fuser.
- **Seqlock snapshot** — the daemon-published shared state block read lock-free by Python (retry on odd sequence).
- **SPSC ring** — single-producer/single-consumer lock-free ring; one per direction between orchestrator and daemon.
- **Stable / streaming / spill zones** — the expert-cache VRAM zones (§3.1).
- **SubComponent bits** — per-expert readiness bitmask (gate/up/down) supporting partial residency (§3.1).
- **Two-zone cache** — the stable+streaming expert-cache design (§3.1).
- **Waterline** — the bounded per-GPU in-flight transfer window in the transfer engine.
