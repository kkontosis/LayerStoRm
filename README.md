# LayerStoRm

**Run frontier-scale MoE language models on a handful of consumer GPUs by streaming experts over PCIe.**

Modern MoE models are almost entirely routed experts — for GLM-5.2, ~408 GB
of routed-expert weights at 4-bit against ~40 GB of everything else — but
each token activates only a few of them, and the router tells you *which*.
LayerStoRm keeps the full expert set in pinned, NUMA-placed host RAM and
fetches the activated experts to VRAM every token — deduplicated,
cost-model-placed across unequal GPUs, and overlapped with compute — so
**PCIe streaming bandwidth, not VRAM size, is the budget**. The full design
is in [`docs/DESIGN.md`](docs/DESIGN.md) (with
[`docs/DESIGN-REVISIONS.md`](docs/DESIGN-REVISIONS.md)); the subsystem tour
is [`docs/INTERNALS.md`](docs/INTERNALS.md).

Measured on one box (512 GB RAM + 2× RTX 5090 + 2× RTX 5080), from the
internal ledgers, each number labeled with its regime:

| Model | Figure | Regime |
|---|---|---|
| GLM-5.3-Flash | 27.0–27.4 tok/s decode @8k | untiered hand champion, banked band |
| GLM-5.3-Flash | 24.5 tok/s decode @8k, 159 tok/s fresh prefill @27k | one-command autoconfig recipe, 1M-context KV tiering, single legs |
| GLM-5.3-Flash | 1M-token context **admitted** | boot-verified windowed admission; a 1M measurement is planned, not claimed |
| GLM-5.2 (744 B) | 10.5 tok/s decode | speculative champion corpus; plain decode ~6.8–7.7 |
| GLM-5.2 (744 B) | 39.6–44.6 tok/s served prefill | HTTP, 672-token through 25k-token prompts |

Active work: predictive expert prefetching (EPM), batched decode, CPU
hybrid decode, AMD and multi-node support.

## Quick start

**Floors first, before you download anything**: serving needs an NVIDIA
SM120 GPU (RTX 5090/5080 class — the only supported hardware; kernels
build as `120f`), and enough host RAM to pin the model's full expert
set — a measured **207.5 GB** for GLM-5.3-Flash, on top of OS and KV
overheads (the reference box carries 512 GB; the weights download alone
is 186 GB). Autoconfig checks the fit and refuses with the binding
constraint named rather than boot into OOM — but check the floors
before the download, not after.

Prerequisites, in one line (Ubuntu-style; CUDA 12.8+ and NCCL 2.20+ come
from [NVIDIA's repos or installer](https://developer.nvidia.com/cuda-downloads)
if apt doesn't have them):

```sh
sudo apt install build-essential cmake git python3 libnuma-dev liburing-dev libnccl2 libnccl-dev
```

Then, with no sudo from here on:

```sh
git clone --recursive https://github.com/kkontosis/LayerStoRm.git
cd LayerStoRm
./scripts/setup.sh    # session-local uv + Python venv + Node.js; checks CUDA; prints next steps
```

(`setup.sh` is checked in rather than piped from the internet — same
one-command convenience, smaller security surface. It installs nothing
system-wide.)

Build (CMake autodetects `nvcc` if `CUDACXX` is unset):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j$(nproc)
.venv/bin/python python/bridge/build_fastbridge.py   # optional Cython ring hot path; the ctypes fallback works but is measurably slower
```

Get a model — the GGUF weights plus the model's HF tokenizer files
(`tokenizer.json` etc.; GGUF-embedded tokenizers are not extracted).
The `hf` CLI is not in `requirements.txt` (it's only needed for this
step — these are plain HTTPS files, any downloader works), so install
it first:

```sh
.toolchain/bin/uv pip install huggingface_hub
.venv/bin/hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir models/GLM-5.3-Flash-GGUF
.venv/bin/hf download zai-org/GLM-5.3-Flash --exclude "*.safetensors" --local-dir test-data/GLM-5.3-Flash
```

Serve — autoconfig derives the whole recipe from the weights and your
hardware, explains every derived field, and refuses (naming the binding
constraint) rather than boot into OOM:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 \
.venv/bin/python python/cli/serve.py --autoconfig \
    --model models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --max-sequence-length 1048576 --max-concurrent 2 \
    --model-name glm-5.3-flash --host 127.0.0.1 --port 8000
```

The two env vars: `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA's device
numbering match the PCI order the derived placement assumes (without it
a mixed 5090/5080 box can swap GPU roles); the deterministic-EP-combine
flag fixes the expert-combine summation order so greedy output is
reproducible run-to-run (the default order follows live expert
placement) — drop it if you don't need reproducibility.

Ask only for the context you need — a 1M ask costs a measured ~10–16% (KV
tiering); smaller asks derive faster untiered configs. Test it:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash", "max_tokens": 128,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

`/v1/completions`, streaming SSE, logprobs, tool calls, reasoning content,
and guided JSON output all work.

## Going deeper

- [`docs/SERVING.md`](docs/SERVING.md) — the full serving procedure: the autoconfig path in detail, the measured calibrate/train/place pipeline, and the manual hand-tuning flow behind the champion recipes.
- [`docs/AUTOCONFIG_MODEL.md`](docs/AUTOCONFIG_MODEL.md) — the config-derivation model: levers, pins, constraint registry, refusal semantics.
- [`docs/BUILDING.md`](docs/BUILDING.md) — full build reference: toolchain, CUDA selection, Python env, the Cython fast bridge, troubleshooting.
- [`docs/INTERNALS.md`](docs/INTERNALS.md) — what's inside, subsystem by subsystem, and the I8 placement model.
- [`docs/DESIGN.md`](docs/DESIGN.md) / [`docs/I8_PLACEMENT_MODEL.md`](docs/I8_PLACEMENT_MODEL.md) — the designs at reimplementation depth.
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — the published work this engine draws on.

## Scope and limitations

**This engine is highly experimental** — an early research-grade release
under active, rapid development: interfaces, configs, and on-disk formats
change without notice, and correctness outside the gated model/hardware
combinations is not guaranteed. Please don't base a production deployment
on it at this stage. Honest boundaries, as of today:

- **Hardware**: PCIe 5.0 consumer GPUs only — NVIDIA SM120 (RTX 5090/5080 class); kernels build as `120f`.
- **Models**: MLA-family MoE only — **GLM-5.3-Flash**, **GLM-5.2**, **DeepSeek-V4-Flash**, **DeepSeek-V3.2** are the bring-up targets with golden-token gates.
- **Single node**; concurrency is modest (concurrency-2 boot-verified on GLM-5.3-Flash; further requests queue with 503 + Retry-After beyond the bound). B>1 batched decode is active work.
- Every performance change is gated on bit-identical golden-token tests.

## License and attributions

MIT License — see [LICENSE.md](LICENSE.md). Third-party notices are
collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); in one line:
the SM120 MLA attention kernels derive from FlashMLA (MIT) and CUTLASS
(BSD-3-Clause), the CPU/GGUF quantized-GEMM and MXFP4/GGUF decode paths
derive from the llama.cpp / ik_llama.cpp lineage (MIT), and parts of the
serving layer follow vLLM and SGLang (Apache-2.0).

## Contributing

Issues and PRs are welcome — measurements from other SM120 boxes
especially. Performance claims in a PR need a number and its regime;
correctness changes need the golden-token gates under `tests/` green.
Start with [`docs/DESIGN.md`](docs/DESIGN.md) and `DEVELOPMENT.md`.
