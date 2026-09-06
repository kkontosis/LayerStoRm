# Serving with LayerStoRm

The complete serving procedure: the autoconfig default path first, then the
manual, hand-tuned flow it automates. Build the engine first
([docs/BUILDING.md](BUILDING.md)).

## 1. The default path: autoconfig from the weights (P-31)

One command derives a full, explained, schema-valid recipe from the weights
and serves it. Boot-verified on GLM-5.3-Flash at 1M context, concurrency 2
(P-31 step 1, 2026-09-06 — the exact call below; the full derivation model
is [docs/AUTOCONFIG_MODEL.md](AUTOCONFIG_MODEL.md)):

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 \
.venv/bin/python python/cli/serve.py --autoconfig \
    --model /srv/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --max-sequence-length 1048576 --max-concurrent 2 \
    --model-name glm-5.3-flash --host 0.0.0.0 --port 8000
```

What you need on disk:

- **The GGUF weights** (any path):
  `hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir <dir>`
- **The HF tokenizer files** (`tokenizer.json`, `tokenizer_config.json`,
  from `zai-org/GLM-5.3-Flash`) in the weights directory, a sibling
  directory, or `test-data/GLM-5.3-Flash/` — GGUF-embedded tokenizers are
  not extracted (TD-SERVE-GGUF-TOKENIZER), so the probe needs real files.
  `--tokenizer-path <dir>` overrides the search.

What the flags mean:

- `--autoconfig --model` derives the recipe from the weights themselves —
  hardware detection is CPU-only, and the derived recipe is persisted next
  to the repo root with an `.explain.md` sidecar naming the constraint
  behind every derived field. Re-boots reuse it while the hardware
  fingerprint matches (`--autoconfig-redetect` forces re-derivation).
- `--max-sequence-length` / `--max-concurrent` become **hard solver pins**
  under `--autoconfig`: the engine is *sized* for that ask (KV pools, KV
  tiering, runtime scratch), not merely HTTP-limited to it. An infeasible
  ask refuses with the binding constraint named, never boots into OOM.
- Optional levers: `--accuracy compact|standard|high|superior` (numerics
  floor) and `--prefer speed|balanced|capacity` (what the fit gives up
  first). See AUTOCONFIG_MODEL.md §2.
- A missing loader calibration self-heals: the recipe derives
  `calibration_mode: full` and the engine measures and writes the artifact
  on first boot. Measured artifacts beside the weights (trained
  calibration, arena placement table) are reused on later derivations.

Ask only for what you need: a 1M ask escalates KV tiering ON, which costs a
measured ~10–13% of decode and ~16% of fresh prefill. Asks the untiered KV
pool can hold (e.g. 2×100k on the reference box) derive tiering OFF and
keep full speed.

### The measured pipeline (calibrate + train + place)

`serve.py --autoconfig` derives and serves. The auto-run pipeline
additionally *measures* your box — hardware calibration, a real-decode
training pass for the placement solver, and a fetch-frequency table for
host-arena placement — and emits a recipe carrying all three artifacts:

```sh
.venv/bin/python python/cli/autoconfigure.py --model <weights>
```

See AUTOCONFIG_MODEL.md §10 for the step-by-step contract (`--reset`,
`--skip-training`, `--skip-placement`, `--pin`, …).

## 2. Talking to the server

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash", "max_tokens": 128,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

`/v1/completions`, streaming SSE, logprobs, tool calls, reasoning content,
and xgrammar-guided `response_format` all work; see `RUN.md` for the serve
recipe notes and the remaining env knobs.

## 3. The manual flow (GLM-5.2 on a 4-GPU box)

The autoconfig path above automates steps 2–4 of what follows. Read on when
you are tuning rather than deploying — this is what the automated flow
runs, on the model the champion recipes were built for.
[`recipes/glm52_serve_champion.json`](../recipes/glm52_serve_champion.json)
is the exact config behind the GLM-5.2 headline numbers;
[`recipes/glm53flash_serve.json`](../recipes/glm53flash_serve.json) is the
GLM-5.3-Flash hand promotion (note: at a 1M ask it is now *behind* the
derivation — it under-reserves the max_seq-scaled runtime headroom; see
AUTOCONFIG_MODEL.md §0).

### 3.1 Get the model and prepack it

Download the GLM-5.2 **UD-Q4_K_XL** GGUF (11 shards) into
`test-data/GLM-5.2-GGUF-Q4_K_XL/`, and the HF tokenizer files
(`tokenizer.json` etc. from `zai-org/GLM-5.2`) into `test-data/GLM-5.2/`:

```sh
hf download unsloth/GLM-5.2-GGUF --include "UD-Q4_K_XL/*" \
    --local-dir test-data/GLM-5.2-GGUF
ln -s GLM-5.2-GGUF/UD-Q4_K_XL test-data/GLM-5.2-GGUF-Q4_K_XL
```

Then prepack the experts once into the engine's DMA-ready on-disk format:

```sh
./build/tools/prepack_experts \
    test-data/config/glm_5_2_gguf.json \
    test-data/GLM-5.2-prepacked
```

This writes ~494 GB; put it on NVMe (the prepacked set is read at
~3.3 GB/s with io_uring O_DIRECT on a Gen3 x4 link — cold arena builds are
disk-bound). Without a prepacked store, GGUF weights are live-prepacked at
boot instead — slower cold boots, same serving.

### 3.2 Calibrate the hardware and train the placement solver

The I8 solver needs to know *your* box. Two steps (both automated by
`autoconfigure --model`, which runs them through the ordinary serve path
instead of the keeper harness):

1. **Calibrate** (measure link/kernel rates): set `gpu_loader.enabled: true`
   with `gpu_loader.calibration_path` pointing at a writable JSON. On first
   run with the file absent, the engine runs a full calibration at init and
   writes it (it self-heals the same way if you delete the file).

   Example:

   ```sh
   export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3
   rm -f test-data/GLM-5.2-GGUF-Q4_K_XL/gpu_loader_calibration_5090x2.json
   LS_LOADER_SHADOW=1 ./build/tests/integration/keeper52_test \
     --gtest_filter='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
   ```

2. **Train** (fit the cost model to real decode timings): run a
   representative decode workload with the solver's prediction dump and the
   perf trace enabled, then bake workload-corrected constants:

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

   (`tools/loader_xray/trainer_apply.py` joins predicted-vs-actual and
   writes the corrected constants; the integration harnesses invoke it
   automatically.) Point `gpu_loader.calibration_path` at the trained file.
   `RUN.md` shows the exact keeper-benchmark form of this flow, including
   the shadow-vs-act A/B.

### 3.3 Fit the arena placement table

The host arena places experts across NUMA banks by a per-(layer, expert)
fetch frequency table (`memory.arena_placement.freq_table`; a table fit to
our champion workload ships in `test-data/placement/`). To fit one to your
workload:

```sh
# collect one traced iteration of your real workload
LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/trace.csv <your run>
# fit (multiple traces accumulate — mix the regimes you serve)
python3 tools/loader_xray/freq_table.py my_freq.csv /tmp/trace.csv
```

Example (ours — one traced champion iteration, then fit; the result ships
as `test-data/placement/glm52_fetch_freq_m3.csv`):

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

Changing the table changes the arena identity: expect **one** cold store
rebuild, then warm attaches keep the placed layout.
`memory.arena_placement.online` (default on) keeps refining placement
during serving.

### 3.4 Prepare the configuration

Everything is one JSON config validated against
[`config/schema.json`](../config/schema.json) —
[`config/config-example.json`](../config/config-example.json) is the
commented starting point, and
[`recipes/glm52_serve_champion.json`](../recipes/glm52_serve_champion.json)
is the exact config behind the headline numbers. Adapt these to your box:

- `hardware.gpus` + `hardware.tp_array` — per-GPU roles: the champion runs
  the two 5090s as the TP attention pair (sharded KV) and all four as
  expert-streaming devices, with `vram_allocation_gb` per GPU.
- `memory.pin_host_expert_pool_sizing` + `memory.cross_node_spill` — arena
  sizing per NUMA bank (the champion pins 0.9 of RAM and spills onto the
  HBM nodes at `fraction_free` 0.8). Size down first; host-OOM during
  registration is the classic first-boot failure.
- `memory.arena_attach` (`enabled` + `persist`) — keeps the pinned store
  alive in a holder process across engine restarts (warm boot ~83–108 s vs
  cold rebuild); `on_conflict` controls what happens when the stored
  identity mismatches.
- `speculation` — the GLM-5.2 champion uses the DSpark arm (γ=15,
  confidence 0.1, NVFP4 draft sharded on the 5080s).
  `speculation.dspark.checkpoint_path` needs a DSpark draft checkpoint,
  which we don't ship — bring your own, or set
  `speculation.enabled: false` and serve plain decode.
- `serving` — host/port, `max_sequence_length`, prefix cache, tool-call
  and reasoning parsers.

### 3.5 Serve the hand recipe

The champion recipe reads three things out of `test-data/`: the GGUF
weights (§3.1), the prepacked experts (§3.1), and a loader calibration,
which must sit **inside the weights directory** —
`gpu_loader.calibration_path` is resolved relative to the weights dir, so
the recipe's `gpu_loader_calibration_ep4x4.json` means
`test-data/GLM-5.2-GGUF-Q4_K_XL/gpu_loader_calibration_ep4x4.json`.
§§3.2–3.3 fit one for your box. To try the recipe immediately without that
detour, copy the ones shipped in `test-data/`:

```sh
cp test-data/gpu_loader_calibration*.json test-data/GLM-5.2-GGUF-Q4_K_XL/
```

They were fit on the reference box (2×RTX 5090 + 2×RTX 5080, HBM NUMA
banks); they will boot anywhere but the placement decisions only reflect
that hardware, so regenerate them for real deployments.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION=bf16 \
.venv/bin/python python/cli/serve.py \
  --config recipes/glm52_serve_champion.json \
  --tokenizer-path test-data/GLM-5.2 \
  --model-name glm-5.2 --host 127.0.0.1 --port 8000
```

## 4. Troubleshooting

Build-side issues (NCCL symbol clashes with torch, CUDA selection) are in
[docs/BUILDING.md](BUILDING.md). Serving-side: the engine's boot log
explains every carve and refusal in the same vocabulary as the autoconfig
explain sidecar; `RUN.md` collects the operational env knobs.
