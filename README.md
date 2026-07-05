# LayerStoRm

**Run frontier-scale MoE language models on a couple of consumer GPUs.**

LayerStoRm is a C++20/CUDA inference engine for Mixture-of-Experts LLMs whose weights vastly exceed GPU memory — think DeepSeek-V3.2-class models (~685 B parameters, ~343 GB of routed experts at 4-bit) on machines with one to four GPUs of the 16–32 GB class and a large host RAM pool. Instead of pinning experts to the CPU or swapping layers synchronously, it **keeps the hot experts cached in VRAM and streams the rest to the GPU over PCIe under tight management** — overlapped with compute, sourced from NUMA-correct pinned memory, placed across GPUs by an explicit cost model, and prefetched just ahead of need.

## Why this works

An MoE model is mostly experts, but each token touches only a few percent of them (8 of 256 per layer), and the router names *which*. That structural sparsity plus PCIe 5.0-class bandwidth is enough to keep expert FFN compute on the GPU, where it belongs — if the engine caches the right experts, keeps every transfer off the critical path, and never moves a byte it doesn't have to. Every subsystem exists to serve that goal. The full argument, mechanisms, and measured performance model live in [`docs/DESIGN.md`](docs/DESIGN.md).

## Highlights

- **Smart GPU expert caching** — a slot-based two-zone VRAM cache per GPU (a *stable* zone for proven-hot experts, a fast-turnover *streaming* zone for the rest) with impact-weighted eviction scoring, two-phase reserve/ready admission, and a single source-of-truth residency map. The streaming zone's memory is shared with the KV cache region, so bursty prefill demand borrows space from the cheapest-to-evict residents instead of the valuable ones.
- **Efficiently managed expert streaming** — per-GPU DMA pipelines with deduplication, priority classes, bandwidth-budgeted planning, and just-in-time dispatch; demand fetches always preempt speculation. Gating-driven lookahead and learned predictors keep the pipeline fed ahead of need.
- **Cost-model expert placement** — a deterministic, microsecond-budget solver (branch-and-bound / subset-partition DP) chooses which GPU fetches and computes each expert per layer, minimizing predicted layer time (transfer roofline, NUMA bank contention, reconciliation, eviction consequence) instead of hashing `expert % gpu_count`.
- **Flexible multi-GPU topologies** — tensor parallelism, expert parallelism, and decode context parallelism (DCP) compose per GPU into micro-configurable setups: e.g. a TP attention pair with DCP-sharded KV plus additional expert-cache-only GPUs, mixed VRAM sizes welcome. Topology is auto-detected and overridable.
- **Single decision authority** — a single-threaded Python orchestrator makes every decision; a C++ daemon thread executes them. They share lock-free SPSC rings and a seqlock state snapshot: no GIL on the hot path, no locks, no races, VRAM accounting that is always truthful.
- **NUMA-native host tier** — per-NUMA-node pinned expert arenas (hundreds of GB via `cudaHostRegister`, unaffected by `RLIMIT_MEMLOCK`), a prepacked on-disk expert format DMA'd without repacking, io_uring NVMe preload, and direct-`pread` cold loads (no mmap fault storms).
- **Well-compressed MLA attention** — custom SM120 (GeForce Blackwell) kernels implementing SnapMLA-style FP8 compressed-latent KV caching (644 B/token/layer with BF16 RoPE — 44% below BF16), with a TurboQuant 4-bit KV backend (386 B/token/layer — 66% below BF16) behind the same interface; DeepSeek Sparse Attention indexing and split-KV decode.
- **Quantization** — NVFP4 (E2M1 + UE8M0 group scales) and FP8 weights; GGUF K-quant kernels for the CPU expert path.
- **Speculative decoding** — MTP-head drafting, prompt lookup, self-speculative reduced-expert/layer-skip drafts; copy-on-write KV forks with metadata-only promotion; draft gating feeds the expert prefetcher across token boundaries.
- **CPU expert fallback** — multi-NUMA tensor-parallel CPU expert device (vectorized NVFP4/GGUF GEMM) for machines where the streaming bet doesn't hold.
- **Correctness as a gate** — deterministic reduction paths and golden-token bit-identity tests guard every performance change; placement and eviction policies are numerically inert by construction.

## Architecture at a glance

```
 Python orchestrator (decisions)          C++ daemon (execution)
 ┌────────────────────────────┐  cmd ring  ┌───────────────────────────┐
 │ predictors → prefetch fuser│──────────▶│ dispatcher · placement    │
 │ transfer & eviction planner│◀──────────│ solver · lifecycle mgr    │
 │ scheduler · statistics     │  cmp ring  │ transfer engine · streams │
 └────────────────────────────┘ + snapshot └─────────────┬─────────────┘
                                                         ▼
     NVMe ──▶ pinned NUMA arenas (host RAM) ──▶ per-GPU expert cache
                                                (stable + streaming zones)
```

Only the daemon touches CUDA, and only in service of orchestrator commands. Prediction modules emit scores, never transfers.

## Requirements

- **GPU**: NVIDIA SM120 (GeForce Blackwell — RTX 5090/5080 class) is the primary target; kernels build as `120f`.
- **Host**: Linux, generous RAM (the host arena caches the expert set — plan for model-sized RAM for best results), NVMe storage, PCIe 5.0 strongly recommended.
- **Toolchain**: CMake 3.25+, CUDA 12.8+, GCC with C++20, NCCL 2.20+, pybind11 (for the Python module). `libnuma` and `liburing` unlock NUMA pinning and the NVMe tier. `nlohmann_json`/`spdlog` are fetched automatically if absent.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run the unit tests:

```sh
./build/tests/unit/layerstorm_unit_tests
```

Build flags, test filtering, and developer workflow are in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Configuration

Everything is driven by a single JSON config validated against [`config/schema.json`](config/schema.json) (field names follow the DeepSeek HF config naming); see [`config/config-example.json`](config/config-example.json). Hardware details — GPU topology, PCIe widths, NUMA layout — are auto-detected and can be overridden. Expert weights are prepacked once into the engine's DMA-ready on-disk format (`--prepack`, or automatically on first run).

## Models

The primary target is the DeepSeek-V3.2 architecture family: MLA attention, DeepSeek Sparse Attention, and large routed-expert MoE.

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Canonical architecture: every core mechanism to reimplementation depth, comparisons with prior systems, performance model, limitations |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Build, test, and contribution workflow |

## Repository layout

```
src/            engine: daemon, memory tiers, transfer, kernels, model loading
python/         orchestrator, HTTP server, tokenizer, pybind11 bindings
config/         JSON schema + examples; generated C++ parser via tools/gen_config.mjs
tests/          unit + integration tests (GoogleTest), offline simulators
bench/          microbenchmarks (H2D bandwidth, NVMe, kernels)
deps/           project kernel collections (attention, expert GEMM, CPU experts)
3rd-party/      vendored build dependencies
docs/           architecture documentation
```

## Status

Active development. The engine decodes DeepSeek V3.2 end-to-end on a 2× RTX 5090 reference machine with bit-checked golden-token correctness; current measured performance and the analytical model of where the approach wins and degrades are maintained in [`docs/DESIGN.md`](docs/DESIGN.md) §10. Single-node only (TP/EP/DCP within one host). Interfaces and configuration are still evolving.

## Acknowledgements

LayerStoRm builds on excellent open work: the MLA attention kernels are adapted from [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (Apache-2.0) and CUTLASS, rebuilt for SM120 with the FP8-KV pipeline; CPU quantized-GEMM kernels are vendored from the `ik_llama` lineage (MIT); and the design draws on the MoE-offloading and MLA research literature (SnapMLA, fMoE, MoE-Infinity, MoE-SpeQ, SP-MoE, MoE-Lightning, and others) cited throughout the internal specs.
