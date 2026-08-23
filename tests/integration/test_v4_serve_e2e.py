"""DeepSeek-V4-Flash serve-stack e2e (ticket I) — real engine over HTTP.

Boots cli.serve.build_stack on the 151 GB Unsloth V4-Flash GGUF with the
ticket-H golden boot recipe (single 5090, vram_gb 30, streaming experts,
no arena/holder contact) + the four deepseek_v4 serving args, then gates
over a real uvicorn wire:

  (a) /v1/completions "The capital of France is" → " Paris..." (the
      ticket-H llama.cpp golden, token 11111, through the whole stack:
      minted GGUF-header tokenizer → teacher-forced prompt feed →
      split/ACT decode → detokenize);
  (b) /v1/chat/completions thinking → STRUCTURAL reasoning split
      (reasoning_content populated with no <think> marker emitted) and
      the V4-mode thinking-off default (content only);
  (d) streaming chunk shape (SSE deltas, reasoning_content routing).

  (c) tool elicitation is opt-in (V4_SERVE_E2E_TOOLS=1): the 292-token
      tools prompt costs ~12 min of teacher-forced prefill
      (TD-V4-PREFILL-PERF); the DSML parser itself is unit-covered and
      was validated live in ticket I (transcript
      scratchpad/ticketI_gateC_tools.json: get_weather {"city":"Paris"},
      finish_reason tool_calls).

Run (opt-in — V4 GGUF + minted tokenizer + one idle >=30 GiB 5090; pick
the device explicitly, e.g. the box's GPU 2):
  V4_SERVE_E2E=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      .venv/bin/python -m pytest tests/integration/test_v4_serve_e2e.py -s

Decode is ~2-4 s/token page-cache streaming — budget ~5 min (plus
~12 min with V4_SERVE_E2E_TOOLS=1).
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_GGUF = pathlib.Path(
    "/srv/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf")
_TOKENIZER_DIR = _GGUF.parent          # minted by tools/mint_v4_hf_tokenizer
_CONFIG = _PROJECT_ROOT / "test-data" / "config" / "deepseek_v4_flash_gguf.json"

PROMPT_TEXT = "The capital of France is"
PROMPT_TOKENS = [671, 6102, 294, 8760, 344]     # ticket-H llama.cpp mint
GOLDEN_NEXT = 11111                             # 'ĠParis'
HTTP_DEADLINE_S = 3600


def _v4_config() -> dict:
    """The shipped V4 config + the ticket-H golden boot recipe
    (deepseek_v4_gguf_golden_test.cpp SetUp)."""
    cfg = json.loads(_CONFIG.read_text())
    cfg["hardware"]["gpus"] = [{"id": 0, "type": "rtx5090", "vram_gb": 30,
                                "pcie_gen": 5, "pcie_width": 16,
                                "numa_node": 0}]
    cfg["hardware"]["tp_array"] = [0]
    cfg["parallelism"]["tensor_parallelism"] = 1
    cfg["memory"]["vram_safety_margin_gb"] = 3.0
    cfg["memory"]["kv_cache"]["page_growth_chunk_tokens"] = 16
    cfg["serving"]["max_concurrent_requests"] = 2
    return cfg


@pytest.mark.skipif(os.environ.get("V4_SERVE_E2E") != "1",
                    reason="set V4_SERVE_E2E=1 to run (151 GB V4 GGUF, "
                           "one >=30 GiB GPU)")
@pytest.mark.skipif(not _GGUF.exists(), reason="V4-Flash GGUF not present")
@pytest.mark.skipif(not (_TOKENIZER_DIR / "tokenizer.json").exists(),
                    reason="minted V4 tokenizer not present — run "
                           "tools/mint_v4_hf_tokenizer.py")
def test_v4_serve_stack_openai_api():
    pytest.importorskip("transformers")
    import httpx

    from cli.serve import ServeOptions, build_stack

    cfg = _v4_config()
    cfg_path = "/tmp/serve_e2e_v4_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    opts = ServeOptions(
        config_path=cfg_path,
        host="127.0.0.1",
        port=0,                          # ephemeral
        model_name="deepseek-v4-flash",
        tokenizer_path="auto",           # minted dir sits next to the GGUF
        tool_call_parser="deepseek_v4",
        enable_auto_tool_choice=True,
        reasoning_parser="deepseek_v4",
        tokenizer_mode="deepseek_v4",    # == the "auto" resolution
    )

    t0 = time.monotonic()
    print(f"\n[v4-serve-e2e] booting V4 serve stack (config={cfg_path}) ...",
          flush=True)
    stack = build_stack(opts)
    loop_thread = threading.Thread(target=stack.orchestrator.run, daemon=True)
    try:
        # ── Architecture-aware boot contract (ticket I) ─────────────────
        orch = stack.orchestrator
        assert orch.bridge.use_far is False
        assert orch.bridge.route_arm == "act"
        assert orch.bridge.first_moe_layer == 0
        assert orch.bridge.decode_timeout_us == 200_000_000
        assert orch.spec.enabled is False
        assert orch.prefix_cache is None
        assert orch._teacher_forced_prefill is True
        assert orch.bridge.vocab_size == 129280      # EngineInfo autodetect
        assert orch.metadata.num_layers == 43
        assert orch.metadata.num_moe_layers == 43    # all-MoE
        # V4-8 metadata: 21 CSA / 20 HCA / 2 SWA-only Flash census.
        att = orch.metadata.v4_attention_types[:43]
        assert (att.count(1), att.count(2), att.count(0)) == (21, 20, 2)

        # Minted tokenizer reproduces the llama.cpp golden ids.
        assert stack.tokenizer.encode(PROMPT_TEXT) == PROMPT_TOKENS
        assert stack.tokenizer.eos_token_ids == (1,)
        assert (stack.tokenizer.think_start_token_id,
                stack.tokenizer.think_end_token_id) == (128821, 128822)

        stack.server.run()
        assert stack.server.wait_ready(timeout=30.0), "uvicorn not ready"
        port = stack.server.bound_port
        assert port
        loop_thread.start()
        base = f"http://127.0.0.1:{port}"
        print(f"[v4-serve-e2e] engine+server up in "
              f"{time.monotonic() - t0:.0f}s at {base}", flush=True)

        with httpx.Client(base_url=base, timeout=HTTP_DEADLINE_S) as client:
            assert client.get("/health").status_code == 200

            # ── (a) golden completion ───────────────────────────────────
            resp = client.post("/v1/completions", json={
                "model": "deepseek-v4-flash",
                "prompt": PROMPT_TEXT,
                "max_tokens": 3,
                "temperature": 0.0,
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            text = body["choices"][0]["text"]
            print(f"[v4-serve-e2e] completion "
                  f"({time.monotonic() - t0:.0f}s): {text!r}", flush=True)
            assert text.startswith(" Paris"), (
                f"expected the golden ' Paris' (token {GOLDEN_NEXT}), "
                f"got {text!r}")
            assert body["usage"]["prompt_tokens"] == len(PROMPT_TOKENS)

            # ── (b) structural reasoning split ──────────────────────────
            resp = client.post("/v1/chat/completions", json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "thinking": True,
                "max_tokens": 8,
                "temperature": 0.0,
            })
            assert resp.status_code == 200, resp.text
            msg = resp.json()["choices"][0]["message"]
            print(f"[v4-serve-e2e] reasoning "
                  f"({time.monotonic() - t0:.0f}s): "
                  f"{msg['reasoning_content']!r}", flush=True)
            assert msg["reasoning_content"], "structural split missing"
            assert "<think>" not in (msg["reasoning_content"] or "")

            # V4-mode default: thinking unspecified → OFF → content only.
            resp = client.post("/v1/chat/completions", json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user",
                              "content": "Reply with exactly: OK"}],
                "max_tokens": 4,
                "temperature": 0.0,
            })
            assert resp.status_code == 200, resp.text
            msg = resp.json()["choices"][0]["message"]
            print(f"[v4-serve-e2e] thinking-off "
                  f"({time.monotonic() - t0:.0f}s): {msg['content']!r}",
                  flush=True)
            assert msg["reasoning_content"] is None
            assert msg["content"]

            # ── (d) streaming chunk shape ───────────────────────────────
            with client.stream("POST", "/v1/chat/completions", json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "thinking": True,
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": True,
            }) as resp:
                assert resp.status_code == 200
                chunks = []
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(json.loads(line[6:]))
            assert chunks, "no SSE chunks"
            assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
            reason_deltas = [c for c in chunks
                             if "reasoning_content"
                             in c["choices"][0]["delta"]]
            assert reason_deltas, "no streaming reasoning deltas"
            assert chunks[-1]["choices"][0].get("finish_reason") in (
                "length", "stop")
            print(f"[v4-serve-e2e] streaming: {len(chunks)} chunks, "
                  f"{len(reason_deltas)} reasoning deltas", flush=True)

            # ── (c) DSML tool elicitation (opt-in: ~12 min prefill) ─────
            if os.environ.get("V4_SERVE_E2E_TOOLS") == "1":
                resp = client.post("/v1/chat/completions", json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user",
                                  "content": "Use get_weather to check the "
                                             "weather in Paris now."}],
                    "thinking": False,
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }],
                })
                assert resp.status_code == 200, resp.text
                choice = resp.json()["choices"][0]
                print(f"[v4-serve-e2e] tools "
                      f"({time.monotonic() - t0:.0f}s): {choice}",
                      flush=True)
                assert choice["finish_reason"] == "tool_calls"
                (tc,) = choice["message"]["tool_calls"]
                assert tc["function"]["name"] == "get_weather"
                assert json.loads(tc["function"]["arguments"]) == {
                    "city": "Paris"}
    finally:
        stack.orchestrator.shutdown()
        if loop_thread.is_alive():
            loop_thread.join(timeout=60.0)
        stack.shutdown()
        try:
            os.remove(cfg_path)
        except OSError:
            pass
    print(f"[v4-serve-e2e] DONE in {time.monotonic() - t0:.0f}s", flush=True)
