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
| Prefill, served | **39.6–44.4 tok/s** | HTTP, 672/1300-token prompts, full champion serving shape (EP4 incl. 5080s, sparse attention + KV tiering live); mini-superchunk strides, default on |
| Context | **TBD** | long-context harness: golden + needle-retrieval + checkpoint/restore under KV tiering. The architecture targets GLM-5.2's 1M positions (allocation-level 1M smoke passes); a max-context measurement is planned, not claimed |
| Boot to serving | ~83–108 s warm attach; cold arena rebuild ~146 s preload at 3.37 GB/s NVMe | persistent arena holder keeps the 494 GB store across engine restarts |

### Milestones

- [x] **VRAM expert caching** — two-zone (stable + streaming) per-GPU cache with impact-weighted eviction
- [x] **I8 heterogeneous device transfer solver** — cost-model expert placement across unequal GPUs
- [x] **DSpark speculative decoding** — lossless draft/verify; 10.5 tok/s champion-corpus decode, +36% over plain at matched trajectory
- [ ] B>1 batched decoding
- [ ] CPU hybrid decoding
- [ ] Predictive fetching / eviction models
- [ ] AMD Radeon AI support
- [ ] Networked multi-node support

Throughput keeps moving: the measured byte-budget model of the B=1 wall puts the
already-identified proposal stack at 10.7–12.4 tok/s with a ~17 tok/s ceiling on
this hardware, and pushing toward it is active work.

## Technologies

- **SnapMLA** — near-lossless FP8 compressed-latent MLA attention kernels, SM120-native (644 B/token/layer — 44% below BF16).
- **TurboQuant** — very efficient 4-bit KV-cache compression behind the same attention interface (386 B/token/layer, ~66% below BF16); composes with the other KV codecs per tier.
- **NUMA-aware host arena** — per-node pinned expert pools sized per bank, including CPU-less HBM-as-RAM nodes; `cudaHostRegister`-pinned (no `RLIMIT_MEMLOCK` ceiling), io_uring O_DIRECT preload, THP-backed registration.
- **PagedAttention** — paged KV with copy-on-write forks and metadata-only promotion of accepted speculative tokens.
- **HiSparse KV tiering** — sparse-attention-guided hot-VRAM / pinned-host KV hierarchy, so long contexts don't have to fit in VRAM.
- **TP with KV sharding** — decode context parallelism: each attention GPU holds a disjoint 1/tp of the sequence.
- **TP with weight sharding** — attention/dense weights split across the TP pair.
- **Expert parallelism (split-EP)** — each expert is computed on the one GPU that caches it; replication is optional caching, never a correctness requirement.
- **Superchunk prefill** *(ours)* — one expert-union fetch stream per layer per 1024-token superchunk instead of per-token fetching, which turns the prefill wall length-independent (measured 8.3× fewer expert bytes, 3.9× wall).
- **Two-zone VRAM expert cache** *(ours, in progress)* — a stable zone for proven-hot experts plus a fast-turnover streaming zone that lends its memory to bursty prefill/KV demand.
- **I8 placement solver** *(ours)* — a greedy + dynamic-programming hybrid that assigns work across devices of different specs (5090s next to 5080s) at the granularity of a single expert evaluation, calibrated to the box and trainable against real decode traces.
- **DSpark speculative decoding** — draft-model speculation with strictly lossless batched verify and KV rewind.
- **Prefix caching** — served prompts fork from cached prefixes (measured 26 s → 3.2 s prefill on a hit), with chain-aware eviction.
- **DMA waterline queues** — per-GPU transfer queues with bounded in-flight DMA and priority staging, keeping every PCIe link saturated without flooding any single GPU.
- **Fast Python / C++ IPC** — lock-free shared-memory command/completion rings between the Python orchestrator and the C++ daemon (Cython fast path, GIL-released waits, pinned IPC region for true-async readbacks); measured orchestration residue is ~0.035 ms per decode round.
- **Persistent RAM loading** — the pinned expert store lives in a holder process and survives engine restarts: warm boots re-attach in ~83–108 s instead of rebuilding ~494 GB.
- **Expert placement statistics** — demand-fetch frequency tables (trace-fit, retrainable) drive host-arena placement, refined online by a placement migrator during serving.
- **Custom SM120 kernel optimizations** — attention, dequant, and MoE kernels tuned for consumer Blackwell (RTX 5090/5080), including split-KV decode, fused gating, and MXFP4/GGUF-native expert paths.
- **Guided decoding + OpenAI-compatible serving** — xgrammar-constrained JSON/grammar output, tool-call and reasoning parsers, streaming SSE (HTTP layer overhead measured at ~0.1–0.2% of a request).

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

Toolchain: **CMake 3.25+, CUDA 12.8+, GCC with C++20, NCCL 2.20+**, pybind11 for
the Python module; `libnuma` and `liburing` unlock NUMA pinning and the NVMe tier
(`nlohmann_json`/`spdlog`/CUTLASS are fetched automatically if absent).

The kernel collections live in sibling repositories, wired in as submodules
(`deps/LayerStoRmKernels`, `deps/LayerStoRmGemmKernels`,
`deps/LayerStoRmExpertKernels`, plus `3rd-party/cutlass`):

```sh
git clone --recursive https://github.com/<org>/LayerStoRm.git
cd LayerStoRm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/tests/unit/layerstorm_unit_tests     # optional sanity
```

For serving you also need a Python 3.10+ venv with `fastapi`, `uvicorn`,
`pydantic` (and `xgrammar` for guided decoding); the engine module is built by
CMake (`LAYERSTORM_BUILD_PYTHON=ON`, the default).

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

   Example (ours — force a fresh full calibration, written weights-adjacent):

   ```sh
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
