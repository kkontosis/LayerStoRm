#!/usr/bin/env bash
# =============================================================================
# serve_guided.sh — production serve boot for the GLM-5.2 champion recipe.
#
#   recipes/serve_guided.sh            # http://0.0.0.0:8000/v1
#
# Brings up the OpenAI-compatible server with guided decoding (structural-tag
# tool calls) and the GLM tool-call / reasoning parsers. Everything that used
# to be an env knob here is now config-first in the recipe JSON:
#
#   compute.dsa_indexer_rewind  keeps DSA sparse attention alive across the
#                               speculative re-feeds (default true)
#   compute.ipc_pin             pins the IPC sideband region (default true)
#
# Override any path without editing this file:
#   LS_SERVE_CONFIG=...   recipe JSON        (default recipes/glm52_serve_champion.json)
#   LS_TOKENIZER_PATH=... tokenizer dir      (default test-data/GLM-5.2)
#   LS_SERVE_HOST/PORT    bind address       (default 0.0.0.0:8000)
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

CONFIG="${LS_SERVE_CONFIG:-$ROOT/recipes/glm52_serve_champion.json}"
TOKENIZER="${LS_TOKENIZER_PATH:-$ROOT/test-data/GLM-5.2}"
HOST="${LS_SERVE_HOST:-0.0.0.0}"
PORT="${LS_SERVE_PORT:-8000}"
PYTHON="${LS_PYTHON:-$ROOT/.venv/bin/python}"

export PYTHONPATH="$ROOT/build/python:$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Shadow solving is a diagnostic; off on the serving path.
export LS_LOADER_SHADOW="${LS_LOADER_SHADOW:-0}"
# Deterministic expert-parallel combine: makes greedy decode reproducible
# run-to-run (the arrival-order atomic combine drifts in the f32 LSB and
# flips close-call argmaxes).
export LAYERSTORM_DETERMINISTIC_EP_COMBINE="${LAYERSTORM_DETERMINISTIC_EP_COMBINE:-1}"
export LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION="${LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION:-bf16}"
# Expert-placement cost model calibrated for this box's EP=4 x TP=4 topology.
export LS_LOADER_CALIB="${LS_LOADER_CALIB:-gpu_loader_calibration_ep4x4.json}"

exec "$PYTHON" python/cli/serve.py \
    --config "$CONFIG" \
    --tokenizer-path "$TOKENIZER" \
    --model-name glm-5.2 \
    --host "$HOST" --port "$PORT" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45
