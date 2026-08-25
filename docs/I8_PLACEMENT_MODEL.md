# The I8 placement model

The cost model the expert-placement solver minimizes, in full. Original work;
the summary equation lives in the [README](../README.md#the-i8-placement-model).


Every MoE layer poses the same question: the router picked `N` experts, some
already resident on some GPU, the rest sitting in host RAM on a particular NUMA
bank — on **which** device should each one be fetched and computed? The devices
are not identical (5090s beside 5080s), the banks are not equidistant, and the
answer changes every token.

I8 answers it by minimizing a hand-derived cost model over the assignment

```
j[·] : {1..N} → {1..M},    j[i] = the device expert i is fetched onto and computed on
G_j  = { i : j[i] = j },   c_j = |G_j|,   P = { j : c_j > 0 }   (participating devices)
```

## The objective

```
T(j[·]) = prep
        + max( device_makespan , bank_egress )
        + recon
        + place_cons_total
        + evict_cons_total
```

The middle term is the point: transfers contend on **two independent resource
classes** — the per-device ingest links and the per-bank egress channels — so
the bottleneck is the larger of the two, not their sum.

**Prep** — NVMe → RAM staging, `j`-independent, so it is a constant offset in
the argmin (kept in the model because the predicted wall must be honest):

```
prep = Σ_i subprep(i),   subprep(i) = 0 if expert i is already in host RAM, else nvme_time(i)
```

**Device makespan** (destination side, grouped by GPU). Transfers run in
parallel across devices but serially within a device's stream, and the batched
grouped-GEMM cannot start until that device's experts have all arrived:

```
subxfer(i) = 0                                                    if cached_gpu[i, j[i]]
           = ncf[tier(numa_bank[i], j[i])] · xfer_speed[j[i]] + xfer_lat[j[i]]   otherwise

R_dest_j        = Σ_{i∈G_j} subxfer(i) + compute_j(c_j)
device_makespan = max_{j∈P} R_dest_j
```

**Bank egress** (source side, grouped by BANK). A bank's memory channel is
shared: every uncached fetch out of bank `b` draws on it regardless of which GPU
it targets, so the floor is the busiest bank:

```
egress(i)   = 0 if cached_gpu[i, j[i]], else ncf[tier(b_i, j[i])] · numa_speed[b_i] + numa_lat[…]
raw_sum_b   = Σ_{ i : numa_bank[i]=b, uncached } egress(i)
g_b         = #{ distinct devices pulling ≥1 uncached expert from bank b }
bank_egress = max_b  raw_sum_b · ( c_b + (1 − c_b)/g_b )
```

`c_b ∈ [0,1]` is the measured **contention factor** of bank `b`. At `c_b = 1`
the channel is strictly serial and the expression collapses to `max_b raw_sum_b`;
at `c_b = 0` it is perfectly parallel and spreading a bank's draw across `g_b`
devices divides the floor by `g_b`. Calibration sets it per box, so the same
code is inert on serial-channel hardware and rewards spreading on parallel
channels — no model change either way.

**Reconciliation** — the TP collective after compute, a slowest-participant
latency plus an additive per-participant term:

```
recon = max_{j∈P} recon_overhead[j] + Σ_{j∈P} recon_added[j]
```

**Consequences** — the two future-token correctives, and the only place the
model looks beyond the current layer:

```
place_cons_total = Σ_{ i : cached_gpu[i, j[i]]=0 } place_cons[i, j[i]]      (may be NEGATIVE)
n_j              = |{ i ∈ G_j : cached_gpu[i, j] = 0 }|
evict_cons_total = Σ_{j∈P} Σ_{u=1..n_j} evict_cons[j, u]                    (≥ 0, convex in n_j)
```

`place_cons` is the one term that can reward an assignment: landing a hot expert
where it will be reused pays off on later tokens. `evict_cons[j, u]` is indexed
by **rank** in device `j`'s descending best-victim list, so the `u`-th eviction
costs at least as much as the `(u−1)`-th — piling fetches onto one GPU gets
progressively more expensive.

## Why compute is not `c · const`

```
compute_j(c) = a_j·c + b_j·⌈c/P_j⌉
```

A per-expert linear cost `a_j` plus a fixed cost `b_j` **per batch**, where `P_j`
is the device's batch width (a grouped-GEMM's saturation width, or a threaded
CPU engine's slot count). The `b_j` overhead amortizes across a full batch, so
there is a real incentive to fill one — and a `(P_j+1)`-th expert opens a new
batch and costs a fresh `+b_j`.

That single step is what gives the model its shape. The batch term pulls toward
**concentrating** experts on a device; the convex eviction term and the parallel
ingest links push toward **spreading** them. The optimum is wherever those
balance, and it moves with the hardware. A linear `c · const` has no batch
structure and systematically over-spreads.

## Solving it

Exactly where exactness is affordable, approximately where it is not, and always
deterministically (lowest-index tie-break):

1. `M^N ≤ 2^22` — full enumeration of the exact objective.
2. `N ≤ 5` — subset-partition dynamic programming, `O(M·3^N)`, exact on the
   makespan and the bank floor.
3. otherwise — greedy prefix: longest-processing-time first freezes the costliest
   experts, and the cheapest few are solved exactly by the DP.

The constants (`xfer_speed`, `numa_speed`, `ncf`, `a_j`, `b_j`, `P_j`, `c_b`, …)
are benchmarked per box and shipped as JSON; `place_cons` and `evict_cons` are
maintained by the orchestrator and can be fit against recorded decode traces.
Steps 2 and 3 of [First steps](#first-steps-glm-52-on-a-4-gpu-box) generate both.
