![LayerStoRm](docs/assets/logo.svg)

**Run frontier-scale MoE LLMs on a handful of VRAM-constrained consumer GPUs by streaming experts over PCIe.**

---

## Currently supported GPUs

| Family | Architecture | Notes |
|---|---|---|
| RTX 50xx | NVIDIA SM120 | 5090 / 5080 tested; kernels build as `120f` |

## Currently supported models (MLA-family MoE)

| Model | Size |
|---|---|
| GLM-5.3-Flash | 320 B (18 B active) |
| GLM-5.2 | 744 B (40 B active) |
| DeepSeek-V4-Flash | 284 B (13 B active) |

## KV quantizations

| Scheme | Bits |
|---|---|
| 1. TurboQuant | 4.03–5.36 |
| 2. FP8 + BF16 RoPE (SnapMLA-style), almost lossless | 8.06–8.94 |

## Weight quantizations

| Format | Bits |
|---|---|
| FP8 | 8 |
| NVFP4 | 4 |
| GGUF — e.g. UD-Q4_K_XL | 2–8 (mixed) |

## TL;DR

**LayerStoRm fits large MoE LLMs, with large context, into much smaller VRAM — efficiently.** MoE models are almost entirely routed experts, but each token activates only a few — and the router tells you which. LayerStoRm keeps the full expert set in pinned host RAM and streams the activated experts to the GPUs every token, overlapped with compute.

- **GPU VRAM is used as a cache** — PCIe streaming bandwidth along with VRAM size, is the budget.
- **Multi-GPU, GPU-only decode**
- **1M context fits on GPU** via KV tiering with RAM offload.
- **Prefix cache** for agentic coding, with RAM-offloaded checkpoints.

## Tested configurations

| Config | GPUs | Host RAM | Interconnect |
|---|---|---|---|
| **#1** | 2× RTX 5090 + 2× RTX 5080 | 512 GB (NUMA-placed) + 64 GB HBM | PCIe 5.0 |

## Measured performance

All numbers measured on Config #1.

| Model | Config | Quant | Weights on disk | Draft | Ctx length | KV cache quant | Prefill (tok/s) | Decode (tok/s) | TTFT |
|---|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | #1 | UD-Q4_K_XL | 186.0 GiB | (WIP) | 1M | FP8 + BF16 RoPE | 159 @27k | 27.0 @0k<br>24.5 @8k | — |
| GLM-5.2 | #1 | UD-Q4_K_XL | 435.2 GiB | dspark γ15 (nvfp4, sharded) | 400k | TQ 4-bit + BF16 RoPE | 28 @8k | 6.8 @0.4k<br>5.3 @8k | — |

## Requirements

- 1 or more supported GPUs, PCIe5 verified link recommended
- Host RAM to pin the model's full expert set: **207.5 GB measured** for GLM-5.3-Flash, plus OS and KV overheads (reference box: 512 GB; the weights download alone is 186 GB)
- CUDA — suggested **13+** — and NCCL 2.20+

## Install

**Prerequisites** (Ubuntu):

```sh
sudo apt install build-essential cmake git python3 libnuma-dev liburing-dev libnccl2 libnccl-dev
```

**CUDA** (suggested 13+; skip if `nvcc --version` is already good) — from [NVIDIA's repos or installer](https://developer.nvidia.com/cuda-downloads) if apt doesn't have it.

**Clone and set up** (no sudo from here on; `setup.sh` is checked in, installs nothing system-wide):

```sh
git clone --recursive https://github.com/kkontosis/LayerStoRm.git
cd LayerStoRm
./scripts/setup.sh
```

**Build:**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j$(nproc)
.venv/bin/python python/bridge/build_fastbridge.py   # optional Cython hot path (faster)
```

**Get a model** (GGUF weights + the model's HF tokenizer/metadata files, downloaded **next to the weights** — the engine resolves the tokenizer, chat template and generation config from the weights directory first; repo `test-data/` is only a loudly-logged fallback):

```sh
.toolchain/bin/uv pip install huggingface_hub
.venv/bin/hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir models/GLM-5.3-Flash-GGUF
.venv/bin/hf download zai-org/GLM-5.3-Flash --exclude "*.safetensors" --local-dir models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL
```

## Serve

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID \
.venv/bin/python python/cli/serve.py --autoconfig \
    --model models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --max-sequence-length 1048576 --max-concurrent 2 \
    --model-name glm-5.3-flash --host 127.0.0.1 --port 8000
```

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash", "max_tokens": 128,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

Notes, kept short:

- Autoconfig derives the whole recipe from the weights and your hardware, explains every derived field, and refuses (naming the binding constraint) rather than boot into OOM.
- The call above is the boot-verified configuration: on a 2×5090 + 2×5080 box it derives snapmla, tp=2, EP4, KV tiering, superchunk stride 2048 — and serves 1M context at concurrency 2. Measured on it: **24.5 tok/s decode** (8k prompt), **159 tok/s prefill**.
- Ask only for the context you need. `1048576` engages KV tiering, which costs ~10–16%; a smaller ask derives a faster untiered config (the hand-tuned 200k champion reaches 27.4 tok/s decode). Untested context lengths are derived the same way but have not been measured.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps CUDA numbering matched to the PCI order the placement assumes (mixed 5090/5080 boxes can otherwise swap GPU roles).
- Add `LAYERSTORM_DETERMINISTIC_EP_COMBINE=1` for run-to-run reproducible greedy output.
- `/v1/completions`, streaming SSE, logprobs, tool calls, reasoning content, and guided JSON output all work.

## Going deeper

- [`docs/SERVING.md`](docs/SERVING.md) — full serving procedure: autoconfig in detail, the calibrate/train/place pipeline, hand-tuning behind the champion recipes
- [`docs/AUTOCONFIG_MODEL.md`](docs/AUTOCONFIG_MODEL.md) — config derivation: levers, pins, constraint registry, refusal semantics
- [`docs/BUILDING.md`](docs/BUILDING.md) — full build reference and troubleshooting
- [`docs/INTERNALS.md`](docs/INTERNALS.md) — subsystem tour and the I8 placement model
- [`docs/DESIGN.md`](docs/DESIGN.md) / [`docs/I8_PLACEMENT_MODEL.md`](docs/I8_PLACEMENT_MODEL.md) — designs at reimplementation depth

## Status and limitations

**Highly experimental** — early research-grade, under rapid development. Interfaces, configs, and on-disk formats change without notice; correctness outside the gated model/hardware combinations is not guaranteed. Not for production.

- Single node; modest concurrency (concurrency-2 boot-verified on GLM-5.3-Flash; further requests queue with 503 + Retry-After)
- Every performance change is gated on bit-identical golden-token tests
- Active work: predictive expert prefetching (EPM), batched decode (B>1), CPU hybrid decode, AMD and multi-node support

## License and attributions

MIT — see [LICENSE.md](LICENSE.md). Third-party notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): SM120 MLA attention kernels derive from FlashMLA (MIT) and CUTLASS (BSD-3-Clause); CPU/GGUF quantized-GEMM and MXFP4/GGUF decode paths derive from the llama.cpp / ik_llama.cpp lineage (MIT); parts of the serving layer follow vLLM and SGLang (Apache-2.0).

## Contributing

Issues and PRs welcome — measurements from other SM120 boxes especially. Performance claims need a number and its regime; correctness changes need the golden-token gates under `tests/` green. Start with [`docs/DESIGN.md`](docs/DESIGN.md) and `DEVELOPMENT.md`.

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
- [HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv.org/abs/2608.07009) — Xie, Huang, Huang, Xu, Ma, Kozyrakis, 2026 — sparse-guided hot-VRAM / pinned-host KV tiering ([SGLang implementation guide](https://docs.sglang.io/docs/advanced_features/hisparse_guide))
- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation — Cheng, Yu, Shao, Li, Xiong et al. — the speculative decode arm
- [PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference](https://arxiv.org/abs/2509.23638) — Yu, Zhang, Dong et al., 2025 — gating lookahead behind the expert prefetch predictor
- [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading](https://arxiv.org/abs/2511.14102) — 2025 — the expert-prediction model
- [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](https://arxiv.org/abs/2510.10302) — 2025 — planning verification's expert transfers from the draft's gating weights
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

DSpark and Training-Free Loosely Speculative Decoding are listed without
links because the copies consulted here carry no canonical URL.

### Thanks

LayerStoRm stands on the shoulders of the open inference ecosystem — for
reference implementations, design ideas, and (where noted in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)) adapted code:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [ktransformers](https://github.com/kvcache-ai/ktransformers)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

And thank you to the authors of the work referenced above. Nearly every
subsystem here started as someone else's published idea; the measurements
in this project's ledgers exist because that work was shared openly.
Errors in adapting it are mine.
