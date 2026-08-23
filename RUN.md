## Serve

Config-first: the champion json carries the engine knobs (placement table,
calibration_path, speculation, prefix cache, arena attach/persist). Env vars
that remain, and why:

- IPC-region pinning (cudaHostRegister, so sideband D2H readbacks are
  true-async DMA — DSP52_BOOST lever 2) is now **config default-on**
  (`compute.ipc_pin`, schema default `true`); `LS_IPC_PIN` remains as an
  env **override only** (`0` = off, non-`0` = on, unset = config).
- `LS_LOADER_SHADOW=0` — python-bridge REEF-arm selection (env-only).
- `LAYERSTORM_DETERMINISTIC_EP_COMBINE=1` (+`_PRECISION=bf16`) — json fields
  exist (`compute.deterministic_ep_combine*`); kept as env only out of habit.
  (Moving them into the json is free: the arena config-identity hashes
  geometry/sizing/spill/placement + the prepacked source, NOT the `compute`
  section — see `ArenaCache::hash_config` — so a `compute` edit never wipes
  the warm store.)
- `CUDA_DEVICE_ORDER`/`CUDA_VISIBLE_DEVICES`/`PYTHONPATH` — process-level.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
LS_LOADER_SHADOW=0 \
LAYERSTORM_DETERMINISTIC_EP_COMBINE=1 LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION=bf16 \
.venv/bin/python python/cli/serve.py \
  --config recipes/glm52_serve_champion.json \
  --tokenizer-path test-data/GLM-5.2 \
  --model-name glm-5.2 --host 127.0.0.1 --port 8000
```

(Public checkouts: `test-data/GLM-5.2` ships without `tokenizer.json` —
download it from the model's HF repo into that directory first.)

## Prepack weights

```
./build/tools/prepack_experts \
    test-data/config/glm_5_2_gguf.json \
    test-data/GLM-5.2-prepacked
```

## keeper test with multiple placement methods

```
cd "$(git rev-parse --show-toplevel)"   # repo root
BIN=build/tests/integration/first_token_test
FILTER='--gtest_filter=RoutedExpertLruFullFitDirectTest.HundredTokenDecodeFetchAndRun_FullFit_EP2'

## Choose one
"$BIN" "$FILTER"                                                                                 # baseline
LS_LOADER_SHADOW=1 LS_LOADER_ACT=1 LS_EVICT_LRU_FALLBACK=1 "$BIN" "$FILTER"                      # plainact
LS_LOADER_SHADOW=1 LS_LOADER_ACT=1 LS_EVICT_LRU_FALLBACK=1 LS_EVICT_DECAY=0.98 "$BIN" "$FILTER"  # plainact
KEEPER_AFFINITY=1 "$BIN" "$FILTER"                                                               # keeper CPU affinity
```

## keeper52 decode benchmark (GLM-5.2 full-fit, HBM cross-node spill)

The standard keeper52 run — 100-token GLM-5.2 GGUF decode, EP=2 on the two 5090s,
`E_CMD_FETCH_AND_RUN_MOE` + sharded DCP + TurboQuant MLA + HiSparse KV tiering, direct-IO
full-fit. Baked-in defaults (no env needed): `LS_LOADER_SHADOW` on, `LS_FAR_ENSURE_RESIDENTS`
on (batched inline-evict decode residency; `=0` restores the legacy per-expert path), and
cross-node arena spill onto RAM nodes 0,1 **plus the CPU-less HBM nodes 4-7** (per-node
`fraction_free` 0.80) — this fits all 19,200 routed experts in the pinned arena (0 cold loads),
the fix that took keeper52 from 1.357 → **2.785 tok/s (2.05×)** vs RAM-only (TD-NUMA-HBM-BANKS
/ TD-KEEPER52-ARENA-RAM, commit c9dc7010).

```
BIN=./build/tests/integration/keeper52_test
FILTER='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3

# Standard run (prints per-token ms + the `── MoE timeline ──` x-ray)
$BIN --gtest_filter="$FILTER"
```

Knobs:
- `KEEPER52_ARENA_FRACTION=<f>` — RAM-node arena fraction of MemTotal (default 0.90; the HBM
  spill nodes are sized `fraction_free` and unaffected). Raising toward 0.98 is host-OOM-risky.
- `LS_FAR_ENSURE_RESIDENTS=0` — revert to the legacy per-expert `ensure_resident` +
  one-shot `apply_far_evictions` make-room (A/B the batched inline-evict path).
- `LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=<csv>` — full deep fetch/DMA/compute x-ray (64M-record
  ring default; per-poll `kPollTick` spam gated off unless `LS_PERF_TRACE_POLLTICKS=1`).
- `LS_KEEPER_DUMP_ALL=1` — print all 100 decoded tokens (for A/B token-identity checks).

Expected steady-state (HBM full-fit): ~359 ms/token (2.785 tok/s), MoE ~281 ms/tok, slow
layers (moe_wait > 10 ms) ≈ 0%. The remaining ~2× to the ~5.5 tok/s pure-H2D-bandwidth floor
(16.5 GB/token ÷ 2× PCIe5) is per-layer AR serialization (attention/compute don't overlap the
fetch), not capacity.

## keeper52 I8 loader-solver calibration (two steps) + the "I8 NEW" (ACT) run

keeper52 now enables the I8 loader/solver by default (`LS_LOADER_SHADOW=1` is the
default; set `LS_LOADER_SHADOW=0` — not `unset` — has no effect, the dispatcher
treats presence as on). The solver computes a balanced expert→GPU placement `j[·]`.
In shadow mode it only LOGS `j[·]` vs the orchestrator's `e%tp`; `LS_LOADER_ACT=1`
makes it actually PLACE miss experts per `j[·]` (the "I8 NEW" variant).

Common:
```
BIN=./build/tests/integration/keeper52_test
FILTER='Keeper52Test.HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'
GGUFDIR=test-data/GLM-5.2-GGUF-Q4_K_XL
BASE=$GGUFDIR/gpu_loader_calibration_5090x2.json          # hardware calibration
TRAINED=$GGUFDIR/gpu_loader_calibration_5090x2.trained.json  # workload-corrected
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3
```

### Step 1 — CALIBRATE (measure hardware rates → write BASE)
Delete any stale file so the engine runs a fresh FULL calibration at init and
writes it weights-adjacent (INV-LOADER-CAL-1 self-heal):
```
rm -f "$BASE"
LS_LOADER_SHADOW=1 $BIN --gtest_filter="$FILTER"
```

### Step 2 — TRAIN (fit the cost-model constants to REAL decode timings → write TRAINED)
Dumps the solver's per-command `j[·]` predictions + captures the perf-trace of the
actual fetch/DMA/compute, then at TearDown runs `tools/loader_xray/trainer_apply.py`
to join predicted-vs-actual and bake a corrected LoaderConstants JSON:
```
LS_LOADER_SHADOW=1 \
  LS_LOADER_SHADOW_DUMP=/tmp/keeper52_shadow.jsonl \
  LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/keeper52_train_trace.csv \
  LS_LOADER_TRAIN_OUT="$TRAINED" LS_LOADER_TRAIN_MODEL=current \
  $BIN --gtest_filter="$FILTER"
```

### Step 3 — I8 NEW / ACT (place experts per the trained solver, with x-ray)
```
LS_LOADER_SHADOW=1 LS_LOADER_ACT=1 LS_LOADER_CALIB="$TRAINED" \
  LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/keeper52_xray_act.csv \
  $BIN --gtest_filter="$FILTER"
```

Baseline for comparison (orchestrator e%tp placement, solver shadow-only):
```
LS_LOADER_SHADOW=1 LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/keeper52_xray_e2.csv \
  $BIN --gtest_filter="$FILTER"
```
Compare the `── MoE timeline ──` "dead / straggler" line: ACT should shrink it
(shadow/e%tp baseline was 451 ms/tok = 72.7% of the fetch wall).


## M3 arena placement table — (re)train the freq CSV

The pinned-arena host placement (`memory.arena_placement.freq_table` in the
config json; env `LS_ARENA_PLACE_FREQ` overrides, `off` disables) is a static
per-(layer,expert) demand-fetch frequency table. It is TRACE-FIT: retrain it
whenever the champion's routing/draft mix shifts (symptom: chunk fetches slow
down while plain steps stay at parity, and wall variance grows).

```
cd "$(git rev-parse --show-toplevel)"   # repo root
# 1. Collect a fetch trace from one champion iteration (any placement arm).
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
LS_PERF_TRACE=1 LS_PERF_TRACE_OUT=/tmp/champ_trace.csv \
KEEPER52_REEF_ORCH=1 \
LS_ARENA_PLACE_FREQ=test-data/placement/glm52_fetch_freq_m3.csv \
DSP52_VB=batched DSP52_OVERLAP=1 DSP52_CONF_THRESH=0.1 \
DSP52_QUANT=nvfp4 DSP52_SHARD=1 \
DSP52_PROMPT=test-data/prompts/glm52_longctx_tokens3.txt DSP52_PROMPT_TOKENS=512 \
DSP52_FORCE_TRAJ=test-data/prompts/dsp52_forced_traj_r1.txt \
./build/tests/integration/dsp52_test \
  --gtest_filter='Dsp52Test.SpeculativeHundredTokenDecodeFetchAndRun_FullFit_EP4_GLM52'

# 2. Fit the table (multiple traces accumulate — mix the regimes you serve).
python3 tools/loader_xray/freq_table.py NEW_TABLE.csv /tmp/champ_trace.csv

# 3. A/B via env before shipping (identity change => ONE cold store rebuild,
#    then warm attaches keep the placed layout):
LS_ARENA_PLACE_FREQ=NEW_TABLE.csv LS_ARENA_PLACE_ONLINE=0 <champion run>

# Ship = replace test-data/placement/glm52_fetch_freq_m3.csv (the GLM-5.2
# config default) or point memory.arena_placement.freq_table at the new file.
```

## License

LayerStoRm is licensed under the Apache License 2.0 — see LICENSE.md.
Third-party notices: THIRD_PARTY_NOTICES.md.
