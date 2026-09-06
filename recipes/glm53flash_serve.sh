#!/usr/bin/env bash
# =============================================================================
# glm53flash_serve.sh — serve boot for GLM-5.3-Flash (glm5_next), GF3.13.
#
#   recipes/glm53flash_serve.sh            # http://0.0.0.0:8000/v1
#
# Single-5090 TP=1 recipe (recipes/glm53flash_serve.json — the GF3.9 proven
# boot shape; TP=2 exists for VRAM headroom but buys decode nothing,
# INV-KDA-TP).  Serving surface:
#   tokenizer_mode glm5_next   reasoning_effort low/high/max (default max;
#                              minimal→low, medium→high, xhigh→max; "none"
#                              and thinking:false are 400s — the GLM-5.3
#                              template has NO thinking-off control),
#                              clear_thinking chat default TRUE (prior-turn
#                              <think> blocks dropped deterministically;
#                              pass clear_thinking:false + reasoning_content
#                              to re-render them)
#   reasoning parser glm45     <think>…</think> split (template pre-seeds
#                              <think>, output begins mid-reasoning)
#   tool-call parser glm47     <tool_call>name<arg_key>… wire format
#   stops                      eos [154820 <|endoftext|>, 154827 <|user|>,
#                              154829 <|observation|>] via autodetect
#   sampling                   unset temperature = greedy champion path;
#                              explicit temperature>0 fills top_p 0.95 from
#                              generation_config.json unless the request
#                              sets top_p (model card: temp 1.0 / top_p
#                              0.95; DeepSWE eval: 0.95 / 1.0)
#
# ⚠ BOX RULE: ONE engine at a time.  A glm5_next boot EVICTS whatever warm
# store the arena holder carries (e.g. the GLM-5.2 champion, ~504 GB shared;
# its next boot then costs ~350 s cold).  Check first:
#   ps -eo args | grep '[s]erve.py'; free -g   # shared ≈ 0 or known-yours
#
# Override any path without editing this file:
#   LS_SERVE_CONFIG=...   recipe JSON   (default recipes/glm53flash_serve.json)
#   LS_SERVE_HOST/PORT    bind address  (default 0.0.0.0:8000)
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

CONFIG="${LS_SERVE_CONFIG:-$ROOT/recipes/glm53flash_serve.json}"
HOST="${LS_SERVE_HOST:-0.0.0.0}"
PORT="${LS_SERVE_PORT:-8000}"
PYTHON="${LS_PYTHON:-$ROOT/.venv/bin/python}"

export PYTHONPATH="$ROOT/build/python:$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"  # PCI order: 0/1 are the 16 GB 5080s; 2 is the first RTX 5090 (the recipe's 30 GB shape)
# Shadow solving is a diagnostic; off on the serving path.
export LS_LOADER_SHADOW="${LS_LOADER_SHADOW:-0}"
# Deterministic expert-parallel combine: greedy decode reproducible
# run-to-run (identity-comparison precondition, CAMPAIGN_DOSSIER §4b).
export LAYERSTORM_DETERMINISTIC_EP_COMBINE="${LAYERSTORM_DETERMINISTIC_EP_COMBINE:-1}"
export LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION="${LAYERSTORM_DETERMINISTIC_EP_COMBINE_PRECISION:-bf16}"

exec "$PYTHON" python/cli/serve.py \
    --config "$CONFIG" \
    --model-name glm-5.3-flash \
    --host "$HOST" --port "$PORT"
