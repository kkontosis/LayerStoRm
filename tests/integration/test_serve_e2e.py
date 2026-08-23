"""Serve-stack e2e (#71) — real engine → successor orchestrator → OpenAI API.

THE serving end-to-end on the CHAMPION stack: cli.serve.build_stack boots
the REAL GLM-5.2 GGUF engine with the dsp52 champion config (EP=4, TQ
attention + sharded KV + tiering, DSpark γ15 + confidence, REEF service
placement, fused FAR sweeps) — the SAME builder the Python bridge test
uses — the REAL HuggingFace tokenizer (test-data/GLM-5.2), a REAL uvicorn
server on an ephemeral port; then a plain HTTP client sends
``/v1/completions`` with the golden TEXT prompt and must get " Paris"
back, and ``/v1/chat/completions`` must return a coherent chat response.

JSON-FIRST CONTRACT (production orientation): every warm-arena and
placement decision rides the CONFIG — ``memory.arena_attach.enabled``,
``memory.arena_placement.freq_table``, prepack + preload,
``gpu_loader.calibration_path``, the dspark block.  This test POPS the
arena override env vars before booting and asserts the built json carries
those fields, so a green run proves the serve path needs none of them.
With the arena holder daemon running, boot warm-attaches the persistent
store (~2 min); without it, first boot cold-builds the store.

Run (opt-in — 4-GPU box + GLM-5.2 assets + DSpark ckpt):
  SERVE_E2E=1 .venv/bin/python -m pytest \
      tests/integration/test_serve_e2e.py -s
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import pytest

# The champion config builder + EP=4 GPU gate from the Python bridge test
# (importing it applies the test-harness env contract: CUDA PCI ordering,
# deterministic EP combine for the golden, IPC pin).
from test_dsp52_bridge import (
    _build_dsp52_config,
    _dsp52_ckpt_dir,
    _ep4_gpus_ok,
)
# Golden prompt/next-token pair from the #91 e2e.
from test_orch_engine_decode import _GGUF, GOLDEN_NEXT, PROMPT_TOKENS

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_TOKENIZER_DIR = _PROJECT_ROOT / "test-data" / "GLM-5.2"

PROMPT_TEXT = "The capital of France is"   # encodes to PROMPT_TOKENS
GEN_TOKENS = 3
HTTP_DEADLINE_S = 1800   # cold store build dominates worst case

# Production adheres to the json config: the arena/placement env overrides
# must NOT be load-bearing for the serve path.
_ARENA_OVERRIDE_ENVS = ("GLM52_ARENA_ATTACH", "LS_ARENA_PLACE_FREQ",
                        "LS_ARENA_PLACE_ONLINE")


@pytest.mark.skipif(os.environ.get("SERVE_E2E") != "1",
                    reason="set SERVE_E2E=1 to run (real 436 GB GGUF engine)")
@pytest.mark.skipif(not _GGUF.exists(), reason="GLM-5.2 GGUF not present")
@pytest.mark.skipif(not (_TOKENIZER_DIR / "tokenizer.json").exists(),
                    reason="GLM-5.2 tokenizer files not present")
@pytest.mark.skipif(not (_dsp52_ckpt_dir() / "config.json").exists(),
                    reason="DSpark checkpoint not present")
@pytest.mark.skipif(not _ep4_gpus_ok(),
                    reason="EP=4 needs 4 idle GPUs (2x >=28 GiB + 2x >=14)")
def test_serve_stack_real_engine_openai_api():
    pytest.importorskip("transformers")
    import httpx

    from cli.serve import ServeOptions, build_stack

    # ── Config: the dsp52 champion shape, json-first ────────────────────
    for k in _ARENA_OVERRIDE_ENVS:
        os.environ.pop(k, None)
    cfg, gamma, _block = _build_dsp52_config()
    # The json must carry the warm-arena contract on its own.
    assert cfg["memory"]["arena_attach"] == {"enabled": True,
                                             "persist": True}
    assert cfg["memory"]["arena_placement"]["freq_table"]
    assert cfg["memory"]["pin_host_expert_pool_preload"] is True
    assert cfg["speculation"]["enabled"] is True
    assert cfg["gpu_loader"]["enabled"] is True
    cfg_path = "/tmp/serve_e2e_glm52_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    opts = ServeOptions(
        config_path=cfg_path,
        host="127.0.0.1",
        port=0,                          # ephemeral — no port conflicts
        model_name="glm-5.2-gguf",
        tokenizer_path=str(_TOKENIZER_DIR),
    )

    t0 = time.monotonic()
    print(f"\n[serve-e2e] booting CHAMPION serve stack (config={cfg_path}, "
          f"gamma={gamma}) ...", flush=True)
    stack = build_stack(opts)
    loop_thread = threading.Thread(target=stack.orchestrator.run, daemon=True)
    try:
        # The successor orchestrator armed the champion decode shape.
        orch = stack.orchestrator
        assert orch.spec.enabled and orch.spec.gamma == gamma
        assert orch.spec.conf_thresh == 0.1
        assert orch.bridge.route_arm == "reef" and orch.bridge.use_far

        # Tokenizer sanity: the REAL tokenizer must reproduce the golden
        # prompt ids — otherwise the " Paris" gate below is meaningless.
        assert stack.tokenizer.encode(PROMPT_TEXT) == PROMPT_TOKENS
        assert orch.metadata.eos_token_ids, "no EOS detected"

        stack.server.run()
        assert stack.server.wait_ready(timeout=30.0), "uvicorn not ready"
        port = stack.server.bound_port
        assert port
        loop_thread.start()
        base = f"http://127.0.0.1:{port}"
        print(f"[serve-e2e] engine+server up in "
              f"{time.monotonic() - t0:.0f}s at {base}", flush=True)

        with httpx.Client(base_url=base, timeout=HTTP_DEADLINE_S) as client:
            # /health + /v1/models over the real wire.
            assert client.get("/health").status_code == 200
            models = client.get("/v1/models").json()
            assert models["data"][0]["id"] == "glm-5.2-gguf"

            # ── THE GATE: text completion returns " Paris" ──────────────
            # temperature 0.0 (explicit) → the champion SPECULATIVE path.
            resp = client.post("/v1/completions", json={
                "model": "glm-5.2-gguf",
                "prompt": PROMPT_TEXT,
                "max_tokens": GEN_TOKENS,
                "temperature": 0.0,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            text = body["choices"][0]["text"]
            print(f"[serve-e2e] completion ({time.monotonic() - t0:.0f}s): "
                  f"{text!r} finish={body['choices'][0]['finish_reason']}",
                  flush=True)
            assert text.startswith(" Paris"), (
                f"expected the golden ' Paris' (token {GOLDEN_NEXT}), "
                f"got {text!r}")
            assert body["usage"]["prompt_tokens"] == len(PROMPT_TOKENS)
            assert body["usage"]["completion_tokens"] == GEN_TOKENS
            st = orch.last_stats
            print(f"[serve-e2e] spec stats: rounds={st.rounds} "
                  f"acc={st.accepted}/{st.proposed} "
                  f"decode={st.decode_wall_ms:.0f} ms", flush=True)

            # ── Chat: template → prefill → decode → detokenize ──────────
            # (no temperature field → unspecified → greedy champion path)
            resp = client.post("/v1/chat/completions", json={
                "model": "glm-5.2-gguf",
                "messages": [{"role": "user",
                              "content": "What is the capital of France? "
                                         "Answer with one word."}],
                "max_tokens": 16,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            msg = body["choices"][0]["message"]
            print(f"[serve-e2e] chat ({time.monotonic() - t0:.0f}s): "
                  f"{msg['content']!r} "
                  f"finish={body['choices'][0]['finish_reason']}", flush=True)
            assert msg["role"] == "assistant"
            assert isinstance(msg["content"], str) and msg["content"].strip()
            assert body["choices"][0]["finish_reason"] in ("stop", "length")
    finally:
        stack.orchestrator.shutdown()
        if loop_thread.is_alive():
            loop_thread.join(timeout=60.0)
        stack.shutdown()
        try:
            os.remove(cfg_path)
        except OSError:
            pass
    print(f"[serve-e2e] DONE in {time.monotonic() - t0:.0f}s", flush=True)
