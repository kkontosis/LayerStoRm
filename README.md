# LayerStoRm

**Run frontier-scale MoE language models on a handful of consumer GPUs by streaming experts over PCIe.**

## How this works

Modern MoE models are almost entirely routed experts — for GLM-5.2, ~408 GB of
routed-expert weights at 4-bit against ~40 GB of everything else — but each token
activates only 8 of 256 experts per layer, and the router tells you *which*.
LayerStoRm exploits that: the full expert set lives in **pinned, NUMA-placed host
RAM** (a ~494 GB prepacked arena for GLM-5.2, DMA-ready on disk, including CPU-less
HBM NUMA nodes used as extra RAM banks), while attention, dense layers, KV cache,
and a two-zone expert cache stay resident in VRAM. Every token, the engine fetches
the activated experts it doesn't already hold — deduplicated, NUMA-correct,
cost-model-placed across GPUs, and overlapped with attention/compute — so **PCIe
streaming bandwidth, not VRAM size, is the budget**. On PCIe 5.0 links (~56 GB/s
per GPU, measured aggregate fetch 105–149 GiB/s across four GPUs) that budget is
enough to keep expert FFNs on the GPU, where they belong. The full **initial** argument and
every mechanism at reimplementation depth is in [`docs/DESIGN.md`](docs/DESIGN.md) —
read together with [`docs/DESIGN-REVISIONS.md`](docs/DESIGN-REVISIONS.md), which
records where implementation experience has since revised it.

LayerStoRm's vision is to optimally combine devices of potentially unrelated
architectures for parallel MoE expert compute, with per-layer precision.

## Current status

> **GLM-5.2 (744 B params, MIT weights) at UD-Q4_K_XL on one box — 512 GB RAM + 64 GB HBM +
> 2× RTX 5090 + 2× RTX 5080 — expert streaming at up to 10.5 tok/s decode,
> 60–95 tok/s prefill, context size TBD.**

Every number below is a banked measurement from the internal ledgers, labeled with
its regime — no projections:

| Figure | Number | Regime (honest label) |
|---|---|---|
| Decode, speculative champion | **10.5 tok/s** (best banked 10.498) | harness free-run on the draft-friendly champion corpus; acceptance 0.56 |
| Decode, plain (no speculation) | ~6.8–7.7 tok/s | 100-token keeper benchmark, harness, bare-seed / corpus bands |
| Prefill, superchunk, C++ harness | **60.5 tok/s** current head (best banked 94.9) | 2876-token prompt, warm arena, golden-token-checked; expert H2D ~32 GB/s sustained |
| Prefill, served | **39.6–44.6 tok/s** | HTTP, 672-token through **25k-token** prompts — sustained at depth (windowed KV admission + batched tiered prefill); full champion serving shape (EP4 incl. 5080s, sparse attention + KV tiering live) |
| Context | **TBD** | long-context harness: golden + needle-retrieval + checkpoint/restore under KV tiering. The architecture targets GLM-5.2's 1M positions (allocation-level 1M smoke passes); a max-context measurement is planned, not claimed |
| Boot to serving | ~83–108 s warm attach; cold arena rebuild ~146 s preload at 3.37 GB/s NVMe | persistent arena holder keeps the 494 GB store across engine restarts |

### Milestones

- [x] **VRAM expert caching** — two-zone (stable + streaming) per-GPU cache with impact-weighted eviction
- [x] **I8 heterogeneous device transfer solver** — cost-model expert placement across unequal GPUs
- [x] **DSpark speculative decoding** — lossless draft/verify; 10.5 tok/s champion-corpus decode, +36% over plain at matched trajectory
- [ ] B>1 batched decoding
- [ ] CPU hybrid decoding
- [ ] **EPM predictive fetching** — hide expert fetch behind compute on the decode critical path via learned next-token expert prediction; the identified route to substantially higher streaming-decode throughput
- [ ] **EPM predictive eviction** — recurrence-prediction cache eviction approaching Belady-class capture (AUC ≥ 0.9 acceptance gate)
- [ ] AMD Radeon AI support
- [ ] Networked multi-node support

Throughput keeps moving: the measured byte-budget model of the B=1 wall puts the
already-identified proposal stack at 10.7–12.4 tok/s with a ~17 tok/s ceiling on
this hardware, and pushing toward it is active work.

## Engine internals

Bold names describe what each part does; an italic parenthetical attributes the
published design it follows (see [References](#references)) or marks work that
originated here. The implementations are ours unless
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records adapted code.

- **FP8 compressed-latent MLA** *(SnapMLA-like)* — near-lossless SM120-native attention kernels: the compressed latent in FP8, the RoPE slice kept BF16 (644 B/token/layer — 44% below BF16).
- **4-bit KV quantization** *(TurboQuant-like)* — very efficient KV-cache compression behind the same attention interface (386 B/token/layer, ~66% below BF16); composes with the other KV codecs per tier.
- **NUMA-aware host arena** — per-node pinned expert pools sized per bank, including CPU-less HBM-as-RAM nodes; `cudaHostRegister`-pinned (no `RLIMIT_MEMLOCK` ceiling), io_uring O_DIRECT preload, THP-backed registration.
- **PagedAttention** — paged KV with copy-on-write forks and metadata-only promotion of accepted speculative tokens.
- **KV tiering** *(HiSparse-like)* — sparse-attention-guided hot-VRAM / pinned-host KV hierarchy, so long contexts don't have to fit in VRAM.
- **TP with KV sharding** — decode context parallelism: each attention GPU holds a disjoint 1/tp of the sequence.
- **TP with weight sharding** — attention/dense weights split across the TP pair.
- **Expert parallelism (split-EP)** — each expert is computed on the one GPU that caches it; replication is optional caching, never a correctness requirement.
- **Superchunk prefill** *(ours)* — one expert-union fetch stream per layer per 1024-token superchunk instead of per-token fetching, which turns the prefill wall length-independent (measured 8.3× fewer expert bytes, 3.9× wall).
- **Two-zone VRAM expert cache** *(ours, in progress)* — a stable zone for proven-hot experts plus a fast-turnover streaming zone that lends its memory to bursty prefill/KV demand.
- **I8 placement solver** *(ours)* — a greedy + dynamic-programming hybrid that assigns work across devices of different specs (5090s next to 5080s) at the granularity of a single expert evaluation, calibrated to the box and trainable against real decode traces.
- **Speculative decoding** *(DSpark-like)* — draft-model speculation with strictly lossless batched verify and KV rewind.
- **Prefix caching** — served prompts fork from cached prefixes (measured 26 s → 3.2 s prefill on a hit), with chain-aware eviction.
- **DMA waterline queues** — per-GPU transfer queues with bounded in-flight DMA and priority staging, keeping every PCIe link saturated without flooding any single GPU.
- **Fast Python / C++ IPC** — lock-free shared-memory command/completion rings between the Python orchestrator and the C++ daemon (Cython fast path, GIL-released waits, pinned IPC region for true-async readbacks); measured orchestration residue is ~0.035 ms per decode round.
- **Persistent RAM loading** — the pinned expert store lives in a holder process and survives engine restarts: warm boots re-attach in ~83–108 s instead of rebuilding ~494 GB.
- **Expert placement statistics** — demand-fetch frequency tables (trace-fit, retrainable) drive host-arena placement, refined online by a placement migrator during serving.
- **Custom SM120 kernel optimizations** — attention, dequant, and MoE kernels tuned for consumer Blackwell (RTX 5090/5080), including split-KV decode, fused gating, and MXFP4/GGUF-native expert paths.
- **Guided decoding + OpenAI-compatible serving** — xgrammar-constrained JSON/grammar output, tool-call and reasoning parsers, streaming SSE (HTTP layer overhead measured at ~0.1–0.2% of a request).

## The I8 placement model *(ours)*

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

### The objective

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

### Why compute is not `c · const`

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

### Solving it

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

## Scope and limitations

**This engine is highly experimental.** It is an early research-grade release under
active, rapid development: interfaces, configs, and on-disk formats change without
notice, and correctness outside the gated model/hardware combinations is not
guaranteed. Please do not base a business or production deployment on it at this
stage.

Honest boundaries, as of today:

- **Hardware**: PCIe 5.0 consumer GPUs only — NVIDIA SM120 (RTX 5090/5080 class); kernels build as `120f`. PCIe 4.0 or datacenter parts are unexplored territory.
- **Models**: MLA-family MoE only — **GLM-5.2**, **DeepSeek-V4-Flash**, **DeepSeek-V3.2** are the bring-up targets with golden-token gates.
- **Single node**, **batch size 1** decode. Concurrent requests queue (bounded FIFO, `serving.max_queued_requests`; overflow answers 503 + Retry-After); prefix cache makes that cheap for shared prompts.
- **Active development** — interfaces and configuration still move. Every performance change is gated on bit-identical golden-token tests.

## Building

> **Dependencies note:** release archives don't carry submodule contents — clone the sibling repos / CUTLASS listed in `.gitmodules` into their paths (or use `git clone --recurse-submodules` once published with resolvable URLs) before configuring.

Toolchain: **CMake 3.25+, CUDA 12.8+, GCC with C++20, NCCL 2.20+, Node.js 18+,
Python 3.10+**; `libnuma` and `liburing` unlock NUMA pinning and the NVMe tier
(`nlohmann_json`/`spdlog`/CUTLASS are fetched automatically if absent).

Node.js is **required to configure**, not optional: the config parser
(`src/config/config_parser.{h,cpp}`) is generated from `config/schema.json` by
`tools/gen_config.mjs`, and CMake looks for `node` at configure time. It needs
no npm packages — only the interpreter.

The kernel collections live in sibling repositories, wired in as submodules
(`deps/LayerStoRmKernels`, `deps/LayerStoRmGemmKernels`,
`deps/LayerStoRmExpertKernels`, plus `3rd-party/cutlass`):

```sh
git clone --recursive https://github.com/kkontosis/LayerStoRm.git
cd LayerStoRm
```

### If CUDA is not your system default

With several CUDA versions installed, point the build at the one you want
before configuring — CMake picks up `CUDACXX`, and the linker needs the
matching runtime libraries:

```sh
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

Skip this if `nvcc` on your `PATH` is already the version you intend to build
with.

### Python environment first

Create the virtualenv **before** configuring CMake — the `layerstorm_engine`
pybind11 module is built against it, and `find_package(pybind11)` resolves
through the packages installed here. These examples use
[uv](https://docs.astral.sh/uv/); `python -m venv` + `pip` works identically.

```sh
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python=3.12
uv pip install -r requirements.txt
```

No `activate` step: `uv pip` picks up `./.venv` from the working directory, and
every command below names the interpreter explicitly, so the flow works in a
plain shell. `source .venv/bin/activate` if you prefer it anyway.

`requirements.txt` covers building and serving. Two more sets are available:
`requirements-dev.txt` (pytest + the end-to-end HTTP client, plus the optional
Cython hot path) and `requirements-tools.txt` (torch/LightGBM for the offline
calibration, placement-solver and expert-prediction tooling under `tools/` —
not needed to serve).

Guided decoding (`--enable-auto-tool-choice`, JSON schema output) pulls
`xgrammar`, and with it torch and triton — several GB. Drop that one line from
`requirements.txt` if you only need plain completions; nothing else on the
serving path imports torch.

### Configure and build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j$(nproc)

# command-ring hot path (Cython, built in place next to the bridge)
.venv/bin/python python/bridge/build_fastbridge.py

./build/tests/unit/layerstorm_unit_tests     # optional sanity
```

The two `-D` hints point CMake at the venv you just made; drop them if
pybind11 is installed system-wide, or pass
`-DLAYERSTORM_BUILD_PYTHON=OFF` to build the C++ engine alone (no serving).

The `build_fastbridge.py` step compiles `bridge._fastbridge`, which removes
the per-command ctypes overhead on the ~320 ring round-trips every decode step
makes. Serving works without it — the bridge falls back to pure ctypes, and
`bridge.ring_bridge.fastbridge_active()` reports which path is live — but the
fallback is measurably slower, so build it unless you have a reason not to.
Re-run it after changing the ring protocol.

## First steps (GLM-5.2 on a 4-GPU box)

### 1. Get the model and prepack it

Download the GLM-5.2 **UD-Q4_K_XL** GGUF (11 shards) into
`test-data/GLM-5.2-GGUF-Q4_K_XL/`, and the HF tokenizer files
(`tokenizer.json` etc. from `zai-org/GLM-5.2`) into `test-data/GLM-5.2/`.
Then prepack the experts once into the engine's DMA-ready on-disk format:

```sh
./build/tools/prepack_experts \
    test-data/config/glm_5_2_gguf.json \
    test-data/GLM-5.2-prepacked
```

This writes ~494 GB; put it on NVMe (the prepacked set is read at ~3.3 GB/s with
io_uring O_DIRECT on a Gen3 x4 link — cold arena builds are disk-bound).

### 2. Calibrate the hardware and train the placement solver

The I8 solver needs to know *your* box. Two steps:

1. **Calibrate** (measure link/kernel rates): set `gpu_loader.enabled: true` with
   `gpu_loader.calibration_path` pointing at a writable JSON. On first run with
   the file absent, the engine runs a full calibration at init and writes it
   (it self-heals the same way if you delete the file).

   Example:

   ```sh
   export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3
   rm -f test-data/GLM-5.2-GGUF-Q4_K_XL/gpu_loader_calibration_5090x2.json
   LS_LOADER_SHADOW=1 ./build/tests/integration/keeper52_test \
     --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
   ```
2. **Train** (fit the cost model to real decode timings): run a representative
   decode workload with the solver's prediction dump and the perf trace enabled,
   then bake workload-corrected constants:

   ```sh
   LS_LOADER_SHADOW=1 \
     LS_LOADER_SHADOW_DUMP=/tmp/shadow.jsonl \
     LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/train_trace.csv \
     LS_LOADER_TRAIN_OUT=calib.trained.json LS_LOADER_TRAIN_MODEL=current \
     <your decode run>
   ```

   Example:

   ```sh
   export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3
   LS_LOADER_SHADOW=1 \
     LS_LOADER_SHADOW_DUMP=/tmp/keeper52_shadow.jsonl \
     LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/keeper52_train_trace.csv \
     LS_LOADER_TRAIN_OUT=test-data/GLM-5.2-GGUF-Q4_K_XL/gpu_loader_calibration_5090x2.trained.json \
     LS_LOADER_TRAIN_MODEL=current \
     ./build/tests/integration/keeper52_test \
     --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
   ```

   (`tools/loader_xray/trainer_apply.py` joins predicted-vs-actual and writes the
   corrected constants; the integration harnesses invoke it automatically.) Point
   `gpu_loader.calibration_path` at the trained file. `RUN.md` shows the exact
   keeper-benchmark form of this flow, including the shadow-vs-act A/B.

### 3. Fit the arena placement table

The host arena places experts across NUMA banks by a per-(layer, expert) fetch
frequency table (`memory.arena_placement.freq_table`; a table fit to our champion
workload ships in `test-data/placement/`). To fit one to your workload:

```sh
# collect one traced iteration of your real workload
LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/trace.csv <your run>
# fit (multiple traces accumulate — mix the regimes you serve)
python3 tools/loader_xray/freq_table.py my_freq.csv /tmp/trace.csv
```

Example (ours — one traced champion iteration, then fit; the result ships as
`test-data/placement/glm52_fetch_freq_m3.csv`):

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/champ_trace.csv \
LS_IPC_PIN=1 KEEPER52_REEF_ORCH=1 \
LS_ARENA_PLACE_FREQ=test-data/placement/glm52_fetch_freq_m3.csv \
DSP52_VB=batched DSP52_OVERLAP=1 DSP52_CONF_THRESH=0.1 \
DSP52_QUANT=nvfp4 DSP52_SHARD=1 \
DSP52_PROMPT=test-data/prompts/glm52_longctx_tokens3.txt DSP52_PROMPT_TOKENS=512 \
DSP52_FORCE_TRAJ=test-data/prompts/dsp52_forced_traj_r1.txt \
./build/tests/integration/dsp52_test \
  --gtest_filter='Dsp52Test.SpeculativeHundredTokenDecodeFetchAndRun_FullFit_EP4_GLM52'
python3 tools/loader_xray/freq_table.py my_freq.csv /tmp/champ_trace.csv
```

Changing the table changes the arena identity: expect **one** cold store rebuild,
then warm attaches keep the placed layout. `memory.arena_placement.online`
(default on) keeps refining placement during serving.

### 4. Prepare the configuration

Everything is one JSON config validated against
[`config/schema.json`](config/schema.json) —
[`config/config-example.json`](config/config-example.json) is the commented
starting point, and
[`recipes/glm52_serve_champion.json`](recipes/glm52_serve_champion.json) is the
exact config behind the headline numbers. Adapt these to your box:

- `hardware.gpus` + `hardware.tp_array` — per-GPU roles: the champion runs the two
  5090s as the TP attention pair (sharded KV) and all four as expert-streaming
  devices, with `vram_allocation_gb` per GPU.
- `memory.pin_host_expert_pool_sizing` + `memory.cross_node_spill` — arena sizing
  per NUMA bank (the champion pins 0.9 of RAM and spills onto the HBM nodes at
  `fraction_free` 0.8). Size down first; host-OOM during registration is the
  classic first-boot failure.
- `memory.arena_attach` (`enabled` + `persist`) — keeps the pinned store alive in
  a holder process across engine restarts (warm boot ~83–108 s vs cold rebuild);
  `on_conflict` controls what happens when the stored identity mismatches.
- `speculation` — the champion uses the DSpark arm (γ=15, confidence 0.1, NVFP4
  draft sharded on the 5080s). `speculation.dspark.checkpoint_path` needs a DSpark
  draft checkpoint, which we don't ship — bring your own, or set
  `speculation.enabled: false` and serve plain decode.
- `serving` — host/port, `max_sequence_length`, prefix cache, tool-call and
  reasoning parsers.

### 5. Serve

The champion recipe reads three things out of `test-data/`. If you jumped
straight here, this is what it needs:

**The GGUF weights**, in `test-data/GLM-5.2-GGUF-Q4_K_XL/` (step 1):

```sh
hf download unsloth/GLM-5.2-GGUF --include "UD-Q4_K_XL/*" \
    --local-dir test-data/GLM-5.2-GGUF
ln -s GLM-5.2-GGUF/UD-Q4_K_XL test-data/GLM-5.2-GGUF-Q4_K_XL
```

**The prepacked experts**, in `test-data/GLM-5.2-prepacked/` (step 1) — the
recipe streams experts from there, not from the GGUF.

**A loader calibration**, which must sit **inside the weights directory**:
`gpu_loader.calibration_path` is resolved relative to the weights dir, so the
recipe's `gpu_loader_calibration_ep4x4.json` means
`test-data/GLM-5.2-GGUF-Q4_K_XL/gpu_loader_calibration_ep4x4.json`. Steps 2-3
fit one for your box. To try the recipe immediately without that detour, copy
the ones shipped in `test-data/`:

```sh
cp test-data/gpu_loader_calibration*.json test-data/GLM-5.2-GGUF-Q4_K_XL/
```

They were fit on the reference box (2×RTX 5090 + 2×RTX 5080, HBM NUMA banks);
they will boot anywhere but the placement decisions only reflect that hardware,
so regenerate them for real deployments.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION=bf16 \
.venv/bin/python python/cli/serve.py \
  --config recipes/glm52_serve_champion.json \
  --tokenizer-path test-data/GLM-5.2 \
  --model-name glm-5.2 --host 127.0.0.1 --port 8000
```

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.2", "max_tokens": 128,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

`/v1/completions`, streaming SSE, logprobs, tool calls, reasoning content, and
xgrammar-guided `response_format` all work; see `RUN.md` for the serve recipe
notes and the remaining env knobs.

#### Troubleshooting

**`undefined symbol: ncclCommResume` (or another `libnccl` symbol) at boot.**
Two NCCLs are in play: the engine module links your *system* `libnccl.so.2`,
while torch ships its own under `.venv/.../nvidia/nccl/lib`. Only one can serve
a process — whichever loads first — so a torch newer than the system NCCL ends
up bound to the older library and cannot find a symbol it needs.

The serving path imports torch before the engine so this cannot happen. If you
hit it in your own script or a test, do the same, or preload torch's copy:

```sh
LD_PRELOAD=$(.venv/bin/python -c "import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], 'lib', 'libnccl.so.2'))") \
  .venv/bin/python your_script.py
```

The newer NCCL is backward compatible, so the engine runs fine against it.
Installing a torch that matches your system NCCL works too, but pins you to it.

## License and attributions

MIT License — see [LICENSE.md](LICENSE.md). Third-party notices are
collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); in one line: the
SM120 MLA attention kernels derive from FlashMLA (MIT) and CUTLASS (BSD-3-Clause),
the CPU/GGUF quantized-GEMM and MXFP4/GGUF decode paths derive from the
llama.cpp / ik_llama.cpp lineage (MIT), and parts of the serving layer follow
vLLM and SGLang (Apache-2.0).

## Contributing

Issues and PRs are welcome — measurements from other SM120 boxes especially.
Performance claims in a PR need a number and its regime; correctness changes need
the golden-token gates under `tests/` green. Start with
[`docs/DESIGN.md`](docs/DESIGN.md) and `DEVELOPMENT.md`.

## References

The design is drawn from published work. These are **influences** — the
implementations here are independent unless
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records that code was
adapted.

### Reflected in the engine

Each of these shaped a subsystem that ships:

- [A Deep-Dive Into the New Flash MLA Kernel](https://github.com/deepseek-ai/FlashMLA) — DeepSeek-AI, 2025 — the SM120 MLA attention kernels derive from FlashMLA (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md))
- [SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining](https://arxiv.org/abs/2602.10718) — Zhang, Su, Hu, Yang et al., 2026 — FP8 compressed-latent decoding with the RoPE slice kept BF16
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh et al., 2025 — the 4-bit KV codec
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023 — paged KV with copy-on-write forks
- [HiSparse: Hierarchical Sparse Attention](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hisparse_guide.md) — SGLang project documentation (design note, not a paper) — sparse-guided hot-VRAM / pinned-host KV tiering
- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation — Cheng, Yu, Shao, Li, Xiong et al. — the speculative decode arm
- [PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference](https://arxiv.org/abs/2509.23638) — Yu, Zhang, Dong et al., 2025 — gating lookahead behind the expert prefetch predictor
- [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading](https://arxiv.org/abs/2511.14102) — 2025 — the expert-prediction model
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 — the MLA + MoE architecture served here
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) — GLM-5 Team, 2026 — the primary target model

### Background

Read while designing the above; not implemented here.

**Attention, KV cache and long context**

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Shah et al., 2024
- [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120) — Bhatia et al., 2025
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201) — 2026
- [KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse](https://arxiv.org/abs/2503.16525) — 2025

**MoE offloading, expert caching and prefetching**

- [MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/abs/2401.14361) — Xue et al., 2024
- [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217) — 2024
- [fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://arxiv.org/abs/2502.05370) — Yu et al., 2025
- [DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs](https://arxiv.org/abs/2602.03495) — 2026
- [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509) — 2026
- [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](https://arxiv.org/abs/2510.10302) — 2025
- [KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models](https://doi.org/10.1145/3731569.3764843) — Chen, Xie, Zhang et al., 2025

**Speculative decoding, early exit and layer skipping**

- [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168) — Zhang et al., 2023
- [Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting](https://arxiv.org/abs/2404.18911) — 2024
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) — Elhoushi, Shrivastava et al., 2024
- [CLaSp: In-Context Layer Skip for Self-Speculative Decoding](https://arxiv.org/abs/2505.24196) — Chen, Shan et al., 2025
- [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061) — Schuster, Fisch et al., 2022
- [Scaling Speculative Decoding with Lookahead Reasoning](https://arxiv.org/abs/2506.19830) — 2025
- [Utility-Driven Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2506.20675) — Saxena, Tsai et al., 2025
- [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052) — McDanel et al., 2026
- Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match

DSpark and Training-Free Loosely Speculative Decoding are listed without links
because the copies consulted here carry no canonical URL.

## Thanks

LayerStoRm stands on the shoulders of the open inference ecosystem — for
reference implementations, design ideas, and (where noted in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)) adapted code:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [ktransformers](https://github.com/kvcache-ai/ktransformers)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

And thank you to the authors of the work in [References](#references). Nearly
every subsystem here started as someone else's published idea; the measurements
in this README exist because that work was shared openly. Errors in adapting it
are mine.
