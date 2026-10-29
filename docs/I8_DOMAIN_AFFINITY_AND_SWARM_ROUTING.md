# Domain-Affinity Request Routing & the I8 Placement Model

This note connects [the I8 placement model](I8_PLACEMENT_MODEL.md) and the
run-to-run determinism contract (`tools/serve_determinism_probe.py`) to
upstream **domain-affinity request routing** and federated MoE serving
(joint collaboration proposal with [Project HiveMind](https://github.com/xaviercallens/HiveMind)).

---

## 1. Where upstream routing enters the I8 objective

Recall the per-layer I8 objective over the assignment
`j[·] : {1..N} → {1..M}` ([I8_PLACEMENT_MODEL.md](I8_PLACEMENT_MODEL.md)):

```
T(j[·]) = prep
        + max( device_makespan , bank_egress )
        + recon
        + place_cons_total
        + evict_cons_total
```

with uncached fetch count on device `j`:

```
n_j              = |{ i ∈ G_j : cached_gpu[i, j] = 0 }|
place_cons_total = Σ_{ i : cached_gpu[i, j[i]]=0 } place_cons[i, j[i]]      (may be NEGATIVE)
evict_cons_total = Σ_{j∈P} Σ_{u=1..n_j} evict_cons[j, u]                    (≥ 0, convex in n_j)
```

Within a single layer, `cached_gpu[i, j]` is an **input** to the solver. Across
a stream of served requests, however, `cached_gpu[i, j]` is determined by the
expert overlap between consecutive tokens and consecutive requests:

1. **Convex eviction penalty (`evict_cons_total`)**: Because `evict_cons[j, u]`
   is indexed by rank in device `j`'s descending best-victim list, the `u`-th
   eviction costs at least as much as the `(u−1)`-th. Reducing the uncached
   fetch count `n_j` by Δ therefore removes the **most expensive** tail
   evictions `u ∈ {n_j − Δ + 1 .. n_j}` first.
2. **Transfer floor (`max(device_makespan, bank_egress)`)**: Every cached hit
   (`cached_gpu[i, j[i]] = 1`) sets both `subxfer(i) = 0` on destination GPU
   `j[i]` and `egress(i) = 0` on source NUMA bank `b_i`.
3. **Domain-conditioned `place_cons[i, j]` and arena placement**: When a
   `LayerStoRm` instance serves a domain-coherent stream (for example, CUDA/C++
   kernel synthesis or Lean 4 formal proofs) rather than an interleaved
   multi-topic FIFO queue, the empirical expert fetch-frequency distribution
   concentrates on a narrower subset of routed experts per layer. Both the
   host-arena frequency table (`python/cli/autoconfigure.py`) and the negative
   `place_cons[i, j]` reuse rewards become sharper.

---

## 2. Quantifying the domain-affinity gain on I8

Suppose each layer activates `N` experts per token and the active request
stream switches between `D` semantic domains. Under FIFO arrival across `D`
orthogonal domains, cross-request expert cache hit probability decays toward
the uniform baseline `p_fifo`. Grouping or routing requests by domain affinity
raises the steady-state GPU cache residency rate on the domain's hot expert
working set to `p_aff > p_fifo`.

Because `evict_cons[j, u]` is convex in `n_j` and `compute_j(c_j) = a_j·c_j + b_j·⌈c_j/P_j⌉`
steps at multiples of the grouped-GEMM saturation width `P_j`, higher
`cached_gpu[i, j]` residency simultaneously:

- lowers expected `n_j = |{ i ∈ G_j : cached_gpu[i, j] = 0 }|`, avoiding the
  steep upper ranks of `evict_cons[j, u]`;
- lets the I8 solver pack cached experts onto the same device up to the batch
  boundary `P_j` (amortizing `+b_j`) without paying `subxfer(i)` or
  `evict_cons[j, u]` for them.

The standalone simulator and bridge in
[`tools/domain_affinity_swarm_bridge.py`](../tools/domain_affinity_swarm_bridge.py)
evaluates the exact I8 equations on multi-request queues to measure the wall
reduction before and after domain-affinity reordering.

---

## 3. Run-to-run determinism for verifiable federated serving

In a federated MoE network where consumer multi-GPU nodes serve requests for an
upstream router, output verification requires that a greedy (`temperature = 0`)
forward pass be reproducible. `LayerStoRm` already isolates and solves the
exact sources of MoE run-to-run non-determinism
(see [`tools/serve_determinism_probe.py`](../tools/serve_determinism_probe.py)):

1. **Placement-independent EP reduction**:
   `LAYERSTORM_DETERMINISTIC_EP_COMBINE=1` (or `compute.deterministic_ep_combine: true`)
   eliminates floating-point reduction order drift when `cached_gpu[i, j]`
   residency shifts between runs (`TD-SERVE-PREFILL-NONDET-RUN-TO-RUN`).
2. **Fixed round shape on speculative arms**:
   `LS_SPEC_GOVERNOR=0` disables wall-clock EMA shape switching in
   `SpecRoundGovernor` (`TD-GLM52-CHAMPION-GREEDY-NONDET-RUN-TO-RUN`).
3. **Cold-share degradation exclusion**:
   `RequestStats.moe_degraded_layers == 0` ensures cold-boot degraded layers
   (`TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY`) are excluded from verification
   attestations.

---

## 4. Using `tools/domain_affinity_swarm_bridge.py`

### Offline I8 domain-affinity benchmark

Run the exact I8 cost model across an interleaved FIFO queue vs. a
domain-affinity reordered queue (no GPU required):

```sh
python3 tools/domain_affinity_swarm_bridge.py --simulate-i8
```

### Launching `serve.py` with the determinism contract and swarm bridge

Start `LayerStoRm` with deterministic EP combine enabled ([SERVING.md](SERVING.md)):

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID \
LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 \
LS_SPEC_GOVERNOR=0 \
.venv/bin/python python/cli/serve.py --autoconfig \
    --model models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --max-sequence-length 131072 --max-concurrent 2 \
    --model-name glm-5.3-flash --host 127.0.0.1 --port 8000
```

Probe local server health, verify the determinism environment contract, and
optionally advertise the node's domain-affinity profile to an OpenAI-compatible
swarm coordinator:

```sh
python3 tools/domain_affinity_swarm_bridge.py \
    --layerstorm-url http://127.0.0.1:8000 \
    --model-name glm-5.3-flash \
    --preferred-domains systems_cpp_cuda,formal_math_lean4
```
