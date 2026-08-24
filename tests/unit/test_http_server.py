"""Tests for the OpenAI-compatible HTTP server."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from orchestrator.orchestrator import InferenceRequest
from orchestrator.types import EngineMetadata, GpuConfig
from server.http_server import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    LayerStoRmServer,
    _RequestTracker,
    _apply_stop_sequences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _metadata(eos: tuple[int, ...] = (2,)) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=1,
        num_moe_layers=4,
        num_experts=8,
        num_layers=6,
        expert_bytes=2_359_296,
        kv_bytes_per_page=4096,
        gpus=(GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                        vram_bytes=32 * 1024**3),),
        eos_token_ids=eos,
    )


def _mock_tokenizer(decode_text: str = "Hello world"):
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    tok.decode.return_value = decode_text
    return tok


def _mock_chat_template():
    tmpl = MagicMock()
    tmpl.render.return_value = "<|begin|>Hello<|end|>"
    return tmpl


def _mock_orchestrator(
    token_ids: list[int] | None = None,
    finish_reason: str = "stop",
    logprob_results: list | None = None,
):
    orch = MagicMock()

    def fake_submit(req: InferenceRequest) -> None:
        if req.on_complete is not None:
            ids = token_ids if token_ids is not None else [100, 101, 102]
            req.on_complete(req.request_id, ids, finish_reason,
                            logprob_results)

    orch.submit_request.side_effect = fake_submit
    return orch


def _make_server(
    *,
    decode_text: str = "Hello world",
    token_ids: list[int] | None = None,
    finish_reason: str = "stop",
    eos: tuple[int, ...] = (2,),
    max_concurrent: int = 32,
    logprob_results: list | None = None,
) -> tuple[LayerStoRmServer, TestClient]:
    server = LayerStoRmServer(
        orchestrator=_mock_orchestrator(token_ids, finish_reason,
                                        logprob_results),
        tokenizer=_mock_tokenizer(decode_text),
        chat_template=_mock_chat_template(),
        metadata=_metadata(eos),
        model_name="test-model",
        max_concurrent=max_concurrent,
    )
    client = TestClient(server.app)
    return server, client


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self):
        _, client = _make_server()
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:

    def test_returns_model_list(self):
        _, client = _make_server()
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "test-model"
        assert body["data"][0]["object"] == "model"


# ---------------------------------------------------------------------------
# Completion endpoint
# ---------------------------------------------------------------------------

class TestCompletionEndpoint:

    def test_valid_string_prompt(self):
        _, client = _make_server(decode_text="generated text")
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "test-model"
        assert body["id"].startswith("cmpl-")
        assert len(body["choices"]) == 1
        assert body["choices"][0]["text"] == "generated text"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_valid_token_list_prompt(self):
        _, client = _make_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": [1, 2, 3, 4],
        })
        assert resp.status_code == 200

    def test_missing_model_422(self):
        _, client = _make_server()
        resp = client.post("/v1/completions", json={
            "prompt": "Hello",
        })
        assert resp.status_code == 422

    def test_max_tokens_propagated(self):
        server, client = _make_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 42,
        })
        assert resp.status_code == 200
        call_args = server._orchestrator.submit_request.call_args
        inf_req = call_args[0][0]
        assert inf_req.max_tokens == 42

    def test_max_tokens_none_becomes_zero(self):
        server, client = _make_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": None,
        })
        assert resp.status_code == 200
        call_args = server._orchestrator.submit_request.call_args
        inf_req = call_args[0][0]
        assert inf_req.max_tokens == 0

    def test_stop_sequence_truncation(self):
        _, client = _make_server(decode_text="Hello world. Stop here. More text.")
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "stop": ". Stop",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "Hello world"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_stream_true_returns_200(self):
        _, client = _make_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "stream": True,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_n_greater_than_1_rejected(self):
        _, client = _make_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
            "n": 2,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Chat completion endpoint
# ---------------------------------------------------------------------------

class TestChatCompletionEndpoint:

    def test_valid_request(self):
        _, client = _make_server(decode_text="Hi there!")
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hi there!"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_template_rendering(self):
        server, client = _make_server()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        server._chat_template.render.assert_called_once()
        call_kwargs = server._chat_template.render.call_args
        assert call_kwargs[0][0] == [{"role": "user", "content": "Hello"}]

    def test_tool_calls_in_response(self):
        tool_output = (
            "Let me check.\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"city": "NYC"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        server = LayerStoRmServer(
            orchestrator=_mock_orchestrator(),
            tokenizer=_mock_tokenizer(tool_output),
            chat_template=_mock_chat_template(),
            metadata=_metadata(),
            model_name="test-model",
            model_type="deepseek_v32",
        )
        client = TestClient(server.app)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        tc = body["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"city": "NYC"}

    def test_missing_messages_422(self):
        _, client = _make_server()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
        })
        assert resp.status_code == 422

    def test_thinking_flag_passed(self):
        server, client = _make_server()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Think!"}],
            "thinking": True,
        })
        assert resp.status_code == 200
        call_kwargs = server._chat_template.render.call_args
        assert call_kwargs[1]["thinking"] is True

    def test_stop_sequences(self):
        _, client = _make_server(decode_text="Answer. STOP more text")
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["STOP"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Answer. "

    def test_empty_messages_rejected(self):
        _, client = _make_server()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [],
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Admission: bounded FIFO queueing (serving contract 2026-08-24 — excess
# requests QUEUE; only queue overflow errors, with 503 + Retry-After)
# ---------------------------------------------------------------------------

class TestAdmissionQueueing:

    def test_queue_overflow_returns_503(self):
        server = LayerStoRmServer(
            orchestrator=MagicMock(),
            tokenizer=_mock_tokenizer(),
            chat_template=_mock_chat_template(),
            metadata=_metadata(),
            model_name="test-model",
            max_concurrent=0,           # nothing ever admitted...
            max_queued_requests=0,      # ...and nothing may wait → 503
        )
        client = TestClient(server.app)
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "1"
        assert resp.json()["error"]["code"] == "server_busy"

    def test_second_request_queues_then_serves(self):
        """Two concurrent requests against a 1-slot server: the second
        WAITS for the slot (no instant error) and completes."""
        import threading as _threading

        release_first = _threading.Event()

        def fake_submit(req: InferenceRequest) -> None:
            def emit():
                release_first.wait(timeout=5.0)
                req.on_complete(req.request_id, [100, 101], "stop", None)
            _threading.Thread(target=emit, daemon=True).start()

        orch = MagicMock()
        orch.submit_request.side_effect = fake_submit
        server = LayerStoRmServer(
            orchestrator=orch,
            tokenizer=_mock_tokenizer(),
            chat_template=_mock_chat_template(),
            metadata=_metadata(),
            model_name="test-model",
            max_concurrent=1,
            max_queued_requests=4,
        )
        client = TestClient(server.app)
        results: dict[int, int] = {}

        def post(i: int) -> None:
            r = client.post("/v1/completions", json={
                "model": "test-model", "prompt": "Hello",
            })
            results[i] = r.status_code

        t1 = _threading.Thread(target=post, args=(1,))
        t1.start()
        # First request must be holding the slot before the second lands.
        deadline = time.monotonic() + 5.0
        while server._admission.active < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert server._admission.active == 1
        t2 = _threading.Thread(target=post, args=(2,))
        t2.start()
        deadline = time.monotonic() + 5.0
        while server._admission.queued < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert server._admission.queued == 1     # queued, NOT bounced
        release_first.set()                       # both now complete
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        assert results == {1: 200, 2: 200}
        assert server._admission.active == 0
        assert server._admission.queued == 0

    def test_admission_queue_fifo_and_disconnect(self):
        """_AdmissionQueue unit semantics: FIFO grant order, overflow
        fail-fast, and a queued waiter abandoning on client disconnect."""
        import asyncio as _asyncio

        from server.http_server import _AdmissionQueue

        async def scenario():
            q = _AdmissionQueue(max_concurrent=1, max_queued=2)
            alive = _make_disconnect_fn(False)
            dead = _make_disconnect_fn(True)

            assert await q.acquire(alive) == "acquired"     # slot held
            # Third-in-line beyond the queue bound fails fast.
            w1 = _asyncio.ensure_future(q.acquire(alive))
            await _asyncio.sleep(0.03)                      # w1 queued
            w2 = _asyncio.ensure_future(q.acquire(alive))
            await _asyncio.sleep(0.03)                      # w2 queued
            assert await q.acquire(alive) == "full"
            # Queue at capacity: even a doomed client is bounced fast.
            assert await q.acquire(dead) == "full"
            # Release → FIFO: w1 gets the slot, then w2.
            q.release()
            assert await w1 == "acquired"
            assert not w2.done()
            q.release()
            assert await w2 == "acquired"
            q.release()
            assert q.active == 0 and q.queued == 0

        _asyncio.run(scenario())

    def test_queued_waiter_disconnect_abandons(self):
        import asyncio as _asyncio

        from server.http_server import _AdmissionQueue

        async def scenario():
            q = _AdmissionQueue(max_concurrent=1, max_queued=2)
            alive = _make_disconnect_fn(False)
            dead = _make_disconnect_fn(True)
            assert await q.acquire(alive) == "acquired"
            res = await q.acquire(dead)          # queued → client gone
            assert res == "disconnected"
            assert q.queued == 0                 # ticket removed
            q.release()
            assert q.active == 0

        _asyncio.run(scenario())


def _make_disconnect_fn(value: bool):
    async def is_disconnected() -> bool:
        return value
    return is_disconnected


# ---------------------------------------------------------------------------
# Finish reason
# ---------------------------------------------------------------------------

class TestFinishReason:

    def test_eos_stop(self):
        _, client = _make_server(finish_reason="stop")
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_max_tokens_length(self):
        _, client = _make_server(finish_reason="length")
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_tool_calls_finish_reason(self):
        tool_output = (
            "```json\n"
            '[{"name": "f", "arguments": {}}]\n'
            "```"
        )
        _, client = _make_server(decode_text=tool_output)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "call"}],
        })
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Usage info
# ---------------------------------------------------------------------------

class TestUsageInfo:

    def test_correct_token_counts(self):
        _, client = _make_server(token_ids=[100, 101, 102])
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        body = resp.json()
        assert body["usage"]["prompt_tokens"] == 3  # encode returns [10, 20, 30]
        assert body["usage"]["completion_tokens"] == 3
        assert body["usage"]["total_tokens"] == 6


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

class TestRequestId:

    def test_unique_ids(self):
        _, client = _make_server()
        resp1 = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "a",
        })
        resp2 = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "b",
        })
        assert resp1.json()["id"] != resp2.json()["id"]

    def test_prefix_per_endpoint(self):
        _, client = _make_server()
        comp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "a",
        })
        chat = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "a"}],
        })
        assert comp.json()["id"].startswith("cmpl-")
        assert chat.json()["id"].startswith("chatcmpl-")


# ---------------------------------------------------------------------------
# EOS stripping
# ---------------------------------------------------------------------------

class TestEosStripping:

    def test_eos_tokens_stripped(self):
        eos_id = 2
        server, client = _make_server(
            token_ids=[100, 101, eos_id],
            eos=(eos_id,),
        )
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Hello",
        })
        assert resp.status_code == 200
        tok = server._tokenizer
        decode_call = tok.decode.call_args[0][0]
        assert eos_id not in decode_call


# ---------------------------------------------------------------------------
# _apply_stop_sequences
# ---------------------------------------------------------------------------

class TestApplyStopSequences:

    def test_single_stop(self):
        text, stopped = _apply_stop_sequences("Hello world. Stop.", ".")
        assert text == "Hello world"
        assert stopped is True

    def test_multiple_stops(self):
        text, stopped = _apply_stop_sequences("abcXdefYghi", ["X", "Y"])
        assert text == "abc"
        assert stopped is True

    def test_no_match(self):
        text, stopped = _apply_stop_sequences("Hello world", ["ZZZ"])
        assert text == "Hello world"
        assert stopped is False

    def test_empty_stop_list(self):
        text, stopped = _apply_stop_sequences("Hello", [])
        assert text == "Hello"
        assert stopped is False


# ---------------------------------------------------------------------------
# Request tracker
# ---------------------------------------------------------------------------

class TestRequestTracker:

    def test_register_and_complete(self):
        tracker = _RequestTracker()
        pr = tracker.register(1)
        assert not pr.event.is_set()
        tracker.complete(1, [10, 20], "stop")
        assert pr.event.is_set()
        assert pr.token_ids == [10, 20]
        assert pr.finish_reason == "stop"

    def test_cancel(self):
        tracker = _RequestTracker()
        pr = tracker.register(1)
        tracker.cancel(1)
        assert pr.event.is_set()
        assert pr.cancelled is True

    def test_complete_unknown_id_no_error(self):
        tracker = _RequestTracker()
        tracker.complete(999, [1], "stop")

    def test_cancel_unknown_id_no_error(self):
        tracker = _RequestTracker()
        tracker.cancel(999)


# ---------------------------------------------------------------------------
# Uvicorn lifecycle (#71 serve entrypoint liveness)
# ---------------------------------------------------------------------------

class TestServerLifecycle:

    def test_run_wait_ready_bound_port_shutdown(self):
        """run() starts uvicorn in a thread; wait_ready() blocks until it
        accepts connections; bound_port resolves the ephemeral port; a
        real HTTP GET round-trips; shutdown() stops the thread."""
        import httpx

        server, _ = _make_server()
        server._host = "127.0.0.1"
        server._port = 0  # ephemeral — no port conflicts in CI
        server.run()
        try:
            assert server.wait_ready(timeout=15.0) is True
            port = server.bound_port
            assert isinstance(port, int) and port > 0
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
            assert resp.status_code == 200
        finally:
            server.shutdown()
        assert not server._server_thread.is_alive()

    def test_bound_port_none_before_start(self):
        server, _ = _make_server()
        assert server.bound_port is None


# ---------------------------------------------------------------------------
# Named parsers: glm47 tool calls + glm45 reasoning (vLLM-parity flags)
# ---------------------------------------------------------------------------

THINK_START = 90
THINK_END = 91

_VOCAB = {
    90: "<think>",
    91: "</think>",
    10: "Weather check",
    20: "<tool_call>",
    21: "get_weather",
    22: "<arg_key>",
    23: "city",
    24: "</arg_key>",
    25: "<arg_value>",
    26: "Paris",
    27: "</arg_value>",
    28: "</tool_call>",
    30: "The answer.",
}

_WEATHER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}]

# reasoning + a complete tool call
_TOOL_TOKENS = [10, 91, 20, 21, 22, 23, 24, 25, 26, 27, 28, 2]
# reasoning + plain content
_REASONING_TOKENS = [10, 91, 30, 2]


def _vocab_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    tok.decode.side_effect = (
        lambda ids: "".join(_VOCAB.get(t, "") for t in ids))
    return tok


def _metadata_with_think(eos: tuple[int, ...] = (2,)) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=1,
        num_moe_layers=4,
        num_experts=8,
        num_layers=6,
        expert_bytes=2_359_296,
        kv_bytes_per_page=4096,
        gpus=(GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                        vram_bytes=32 * 1024**3),),
        eos_token_ids=eos,
        think_start_token_id=THINK_START,
        think_end_token_id=THINK_END,
    )


def _make_parser_server(
    *,
    token_ids: list[int],
    tool_call_parser: str = "glm47",
    enable_auto_tool_choice: bool = True,
    reasoning_parser: str = "glm45",
) -> tuple[LayerStoRmServer, TestClient]:
    server = LayerStoRmServer(
        orchestrator=_mock_orchestrator(token_ids, "stop"),
        tokenizer=_vocab_tokenizer(),
        chat_template=_mock_chat_template(),
        metadata=_metadata_with_think(),
        model_name="test-model",
        tool_call_parser=tool_call_parser,
        enable_auto_tool_choice=enable_auto_tool_choice,
        reasoning_parser=reasoning_parser,
    )
    return server, TestClient(server.app)


class TestNamedParserBoot:

    def test_unknown_tool_parser_raises(self):
        with pytest.raises(ValueError, match="unknown tool call parser"):
            _make_parser_server(token_ids=[30], tool_call_parser="bogus")

    def test_unknown_reasoning_parser_raises(self):
        with pytest.raises(ValueError, match="unknown reasoning parser"):
            _make_parser_server(token_ids=[30], reasoning_parser="bogus")

    def test_auto_tool_choice_requires_parser(self):
        with pytest.raises(ValueError, match="requires a tool_call_parser"):
            _make_parser_server(
                token_ids=[30], tool_call_parser="",
                enable_auto_tool_choice=True)


class TestGlm47NonStreaming:

    def _chat(self, client, **overrides):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": _WEATHER_TOOLS,
        }
        body.update(overrides)
        return client.post("/v1/chat/completions", json=body)

    def test_tool_call_parsed(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client)
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        msg = choice["message"]
        assert msg["reasoning_content"] == "Weather check"
        assert msg["content"] is None
        (tc,) = msg["tool_calls"]
        assert tc["id"] == "call_0"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    def test_reasoning_split_no_tools(self):
        _, client = _make_parser_server(token_ids=_REASONING_TOKENS)
        resp = self._chat(client, tools=None)
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        msg = choice["message"]
        assert msg["reasoning_content"] == "Weather check"
        assert msg["content"] == "The answer."
        assert msg["tool_calls"] is None

    def test_thinking_false_no_split(self):
        _, client = _make_parser_server(token_ids=[30, 2])
        resp = self._chat(client, tools=None, thinking=False)
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] is None
        assert msg["content"] == "The answer."

    def test_tool_choice_none_bypasses_parser(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice="none")
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        msg = choice["message"]
        assert msg["tool_calls"] is None
        assert "<tool_call>" in msg["content"]

    def test_no_tools_no_parse(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tools=None)
        msg = resp.json()["choices"][0]["message"]
        assert msg["tool_calls"] is None
        assert "<tool_call>" in msg["content"]

    def test_named_tool_choice_400(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice={
            "type": "function", "function": {"name": "get_weather"}})
        assert resp.status_code == 400
        assert "named tool_choice" in resp.json()["error"]["message"]

    def test_required_tool_choice_400(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice="required")
        assert resp.status_code == 400

    def test_malformed_tool_choice_422(self):
        _, client = _make_parser_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice="sometimes")
        assert resp.status_code == 422

    def test_auto_without_enable_flag_400(self):
        _, client = _make_parser_server(
            token_ids=_TOOL_TOKENS, enable_auto_tool_choice=False)
        resp = self._chat(client, tool_choice="auto")
        assert resp.status_code == 400
        assert "enable-auto-tool-choice" in resp.json()["error"]["message"]

    def test_auto_disabled_tools_absent_ok(self):
        _, client = _make_parser_server(
            token_ids=_REASONING_TOKENS, enable_auto_tool_choice=False)
        resp = self._chat(client, tools=None)
        assert resp.status_code == 200

    def test_legacy_model_type_parser_still_works(self):
        # No named parser configured: pre-flag behavior (model_type
        # registry parser, always-on) is preserved.
        server = LayerStoRmServer(
            orchestrator=_mock_orchestrator(_TOOL_TOKENS, "stop"),
            tokenizer=_vocab_tokenizer(),
            chat_template=_mock_chat_template(),
            metadata=_metadata_with_think(),
            model_name="test-model",
            model_type="glm_moe_dsa",
        )
        client = TestClient(server.app)
        resp = self._chat(client, tools=None)
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == \
            "get_weather"


# ---------------------------------------------------------------------------
# Guided decoding (TD-SERVE-NAMED-TOOL-CHOICE): named tool_choice /
# "required" are now FUNCTIONAL — the server compiles a grammar spec and
# attaches a per-request GuidedState to the InferenceRequest.
# ---------------------------------------------------------------------------


class _StubGuidedState:
    pass


class _StubGuidedGrammar:
    def new_state(self):
        return _StubGuidedState()


class _StubGuidedManager:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []

    def build(self, tools, tool_choice, reasoning, model="glm_4_7"):
        from server.guided import GuidedError
        self.calls.append((tools, tool_choice, reasoning, model))
        if self.fail:
            raise GuidedError("guided grammar construction failed: boom")
        return _StubGuidedGrammar()


def _make_guided_server(
    *,
    token_ids: list[int],
    manager: "_StubGuidedManager | None" = None,
    finish_reason: str = "tool_calls",
) -> tuple[LayerStoRmServer, TestClient, "_StubGuidedManager"]:
    mgr = manager if manager is not None else _StubGuidedManager()
    server = LayerStoRmServer(
        orchestrator=_mock_orchestrator(token_ids, finish_reason),
        tokenizer=_vocab_tokenizer(),
        chat_template=_mock_chat_template(),
        metadata=_metadata_with_think(),
        model_name="test-model",
        tool_call_parser="glm47",
        enable_auto_tool_choice=True,
        reasoning_parser="glm45",
        guided_manager=mgr,
    )
    return server, TestClient(server.app), mgr


class TestGuidedToolChoice:

    def _chat(self, client, **overrides):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": _WEATHER_TOOLS,
        }
        body.update(overrides)
        return client.post("/v1/chat/completions", json=body)

    _NAMED = {"type": "function", "function": {"name": "get_weather"}}

    def test_named_tool_choice_now_functional(self):
        server, client, mgr = _make_guided_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice=self._NAMED)
        assert resp.status_code == 200, resp.text
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        (tc,) = choice["message"]["tool_calls"]
        assert tc["function"]["name"] == "get_weather"
        # The grammar spec was compiled for this request (reasoning=True:
        # thinking unspecified → the template pre-seeds <think>; grammar
        # model follows the configured tool parser — glm47 here).
        assert mgr.calls == [(_WEATHER_TOOLS, self._NAMED, True, "glm_4_7")]
        # The InferenceRequest carried the per-request guided state.
        req = server._orchestrator.submit_request.call_args[0][0]
        assert isinstance(req.guided, _StubGuidedState)

    def test_required_tool_choice_now_functional(self):
        server, client, mgr = _make_guided_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice="required")
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
        assert mgr.calls[0][1] == "required"

    def test_thinking_false_builds_reasoning_false_grammar(self):
        _, client, mgr = _make_guided_server(token_ids=[30, 2])
        resp = self._chat(client, tool_choice=self._NAMED, thinking=False)
        assert resp.status_code == 200
        assert mgr.calls == [(_WEATHER_TOOLS, self._NAMED, False, "glm_4_7")]

    def test_named_function_not_in_tools_400(self):
        _, client, _ = _make_guided_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tool_choice={
            "type": "function", "function": {"name": "nope"}})
        assert resp.status_code == 400
        assert "not in tools" in resp.json()["error"]["message"]

    def test_required_without_tools_400(self):
        _, client, _ = _make_guided_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client, tools=None, tool_choice="required")
        assert resp.status_code == 400

    def test_grammar_build_failure_400(self):
        _, client, _ = _make_guided_server(
            token_ids=_TOOL_TOKENS, manager=_StubGuidedManager(fail=True))
        resp = self._chat(client, tool_choice=self._NAMED)
        assert resp.status_code == 400
        assert "guided" in resp.json()["error"]["message"]

    def test_unconstrained_requests_carry_no_guided_state(self):
        server, client, mgr = _make_guided_server(token_ids=_TOOL_TOKENS)
        resp = self._chat(client)          # tool_choice absent → auto
        assert resp.status_code == 200
        assert mgr.calls == []
        req = server._orchestrator.submit_request.call_args[0][0]
        assert req.guided is None


# ---------------------------------------------------------------------------
# deepseek_v4 serving args (ticket I): DSML tool parser + structural
# reasoning parser + tokenizer_mode thinking defaults + reasoning_config
# passthrough + guided grammar model selection.
# ---------------------------------------------------------------------------

_DSML = "｜DSML｜"
V4_THINK_START = 92
V4_THINK_END = 93

_V4_VOCAB = {
    V4_THINK_START: "<think>",
    V4_THINK_END: "</think>",
    40: "Checking the weather.",
    41: f"<{_DSML}tool_calls>\n",
    42: f'<{_DSML}invoke name="get_weather">\n',
    43: f'<{_DSML}parameter name="city" string="true">Paris'
        f"</{_DSML}parameter>\n",
    44: f"</{_DSML}invoke>\n",
    45: f"</{_DSML}tool_calls>",
    46: "The answer.",
}

# structural reasoning (no <think> emitted) + a complete DSML call
_V4_TOOL_TOKENS = [40, V4_THINK_END, 41, 42, 43, 44, 45, 2]
_V4_REASONING_TOKENS = [40, V4_THINK_END, 46, 2]


def _v4_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    tok.decode.side_effect = (
        lambda ids: "".join(_V4_VOCAB.get(t, "") for t in ids))
    return tok


def _v4_metadata(eos: tuple[int, ...] = (2,)) -> EngineMetadata:
    return EngineMetadata(
        num_gpus=1, num_moe_layers=4, num_experts=8, num_layers=6,
        expert_bytes=2_359_296, kv_bytes_per_page=4096,
        gpus=(GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                        vram_bytes=32 * 1024**3),),
        eos_token_ids=eos,
        think_start_token_id=V4_THINK_START,
        think_end_token_id=V4_THINK_END,
    )


def _make_v4_server(
    *,
    token_ids: list[int],
    tokenizer_mode: str = "deepseek_v4",
    reasoning_config: dict | None = None,
    guided_manager=None,
) -> tuple[LayerStoRmServer, TestClient]:
    server = LayerStoRmServer(
        orchestrator=_mock_orchestrator(token_ids, "stop"),
        tokenizer=_v4_tokenizer(),
        chat_template=_mock_chat_template(),
        metadata=_v4_metadata(),
        model_name="test-model",
        tool_call_parser="deepseek_v4",
        enable_auto_tool_choice=True,
        reasoning_parser="deepseek_v4",
        reasoning_config=reasoning_config,
        tokenizer_mode=tokenizer_mode,
        guided_manager=guided_manager,
    )
    return server, TestClient(server.app)


class TestDeepSeekV4Serving:

    def _chat(self, client, **overrides):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": _WEATHER_TOOLS,
        }
        body.update(overrides)
        return client.post("/v1/chat/completions", json=body)

    def test_dsml_tool_call_parsed_with_structural_reasoning(self):
        _, client = _make_v4_server(token_ids=_V4_TOOL_TOKENS)
        resp = self._chat(client, thinking=True)
        assert resp.status_code == 200, resp.text
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        msg = choice["message"]
        # Structural boundary: everything before </think> is reasoning
        # even though no <think> marker was emitted.
        assert msg["reasoning_content"] == "Checking the weather."
        assert msg["content"] is None
        (tc,) = msg["tool_calls"]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    def test_v4_mode_thinking_defaults_off(self):
        # V4 template defaults thinking OFF: an unspecified `thinking`
        # must NOT trigger the reasoning-first split (unlike GLM mode).
        _, client = _make_v4_server(token_ids=[46, 2])
        resp = self._chat(client, tools=None)
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] is None
        assert msg["content"] == "The answer."

    def test_v4_mode_thinking_true_splits(self):
        _, client = _make_v4_server(token_ids=_V4_REASONING_TOKENS)
        resp = self._chat(client, tools=None, thinking=True)
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "Checking the weather."
        assert msg["content"] == "The answer."

    def test_v4_mode_reasoning_effort_none_disables_thinking(self):
        _, client = _make_v4_server(token_ids=[46, 2])
        resp = self._chat(client, tools=None, thinking=True,
                          reasoning_effort="none")
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] is None
        assert msg["content"] == "The answer."

    def test_v4_mode_render_kwargs_normalized(self):
        server, client = _make_v4_server(token_ids=[46, 2])
        self._chat(client, tools=None, thinking=True,
                   reasoning_effort="xhigh")
        kwargs = server._chat_template.render.call_args[1]
        assert kwargs["thinking"] is True
        assert kwargs["reasoning_effort"] == "max"
        self._chat(client, tools=None, thinking=True,
                   reasoning_effort="low")
        kwargs = server._chat_template.render.call_args[1]
        assert kwargs["reasoning_effort"] == "high"

    def test_legacy_mode_thinking_defaults_on(self):
        # Same server but legacy tokenizer mode: unspecified thinking
        # keeps the historical reasoning-first behavior.
        _, client = _make_v4_server(
            token_ids=_V4_REASONING_TOKENS, tokenizer_mode="hf")
        resp = self._chat(client, tools=None)
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "Checking the weather."

    def test_reasoning_config_marker_override(self):
        # Override forces the text path with the configured markers.
        server, _ = _make_v4_server(
            token_ids=_V4_REASONING_TOKENS,
            reasoning_config={"reasoning_start_str": "",
                              "reasoning_end_str": "<STOP>"})
        p = server._reasoning_parser
        assert p.end_str == "<STOP>"
        assert not p.has_token_ids

    def test_guided_model_follows_v4_parser(self):
        mgr = _StubGuidedManager()
        server, client = _make_v4_server(
            token_ids=_V4_TOOL_TOKENS, guided_manager=mgr)
        named = {"type": "function", "function": {"name": "get_weather"}}
        resp = self._chat(client, tool_choice=named, thinking=True)
        assert resp.status_code == 200, resp.text
        assert mgr.calls == [(_WEATHER_TOOLS, named, True, "deepseek_v4")]

    def test_guided_reasoning_false_by_default_in_v4_mode(self):
        # thinking unspecified in V4 mode → grammar engages at position 0.
        mgr = _StubGuidedManager()
        _, client = _make_v4_server(
            token_ids=_V4_TOOL_TOKENS, guided_manager=mgr)
        named = {"type": "function", "function": {"name": "get_weather"}}
        self._chat(client, tool_choice=named)
        assert mgr.calls[0][2] is False


# ---------------------------------------------------------------------------
# Logprobs (TD-ORCH-LOGPROBS): OpenAI response shapes from per-token
# StepLogprobs delivered via on_complete.
# ---------------------------------------------------------------------------

def _step_lp(tid: int, lp: float, tops: list[tuple[int, float]]):
    from orchestrator.types import StepLogprobs, TokenLogprob
    return StepLogprobs(token=TokenLogprob(tid, lp),
                        top_logprobs=tuple(TokenLogprob(t, v)
                                           for t, v in tops))


def _lp_vocab_tokenizer(vocab: dict[int, str]):
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    tok.decode.side_effect = (
        lambda ids: "".join(vocab.get(t, f"<{t}>") for t in ids))
    return tok


def _make_lp_server(token_ids, logprob_results, vocab):
    server = LayerStoRmServer(
        orchestrator=_mock_orchestrator(token_ids, "stop", logprob_results),
        tokenizer=_lp_vocab_tokenizer(vocab),
        chat_template=_mock_chat_template(),
        metadata=_metadata((2,)),
        model_name="test-model",
    )
    return server, TestClient(server.app)


_LP_VOCAB = {100: "Hello", 101: " world", 102: "!", 7: "alt", 2: "<eos>"}
_LP_STEPS = [
    _step_lp(100, -0.1, [(100, -0.1), (7, -2.0)]),
    _step_lp(101, -0.2, [(101, -0.2), (7, -1.5)]),
    _step_lp(102, -0.3, [(102, -0.3), (7, -1.1)]),
]


class TestCompletionLogprobs:

    def test_logprobs_block_shape(self):
        _, client = _make_lp_server([100, 101, 102], _LP_STEPS, _LP_VOCAB)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "logprobs": 2,
        })
        assert resp.status_code == 200
        lp = resp.json()["choices"][0]["logprobs"]
        assert lp is not None
        assert lp["tokens"] == ["Hello", " world", "!"]
        assert lp["token_logprobs"] == [-0.1, -0.2, -0.3]
        assert lp["text_offset"] == [0, 5, 11]
        assert len(lp["top_logprobs"]) == 3
        assert lp["top_logprobs"][0] == {"Hello": -0.1, "alt": -2.0}

    def test_logprobs_off_serves_null(self):
        _, client = _make_lp_server([100, 101, 102], None, _LP_VOCAB)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi",
        })
        assert resp.json()["choices"][0]["logprobs"] is None
        # And the request threaded logprobs=None into the orchestrator.

    def test_logprobs_propagated_to_request(self):
        server, client = _make_lp_server([100], _LP_STEPS[:1], _LP_VOCAB)
        client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "logprobs": 5,
        })
        inf_req = server._orchestrator.submit_request.call_args[0][0]
        assert inf_req.logprobs == 5

    def test_eos_strip_keeps_alignment(self):
        # Trailing EOS (2) is stripped from the served text; the extra
        # step entry must not shift/appear in the logprobs block.
        steps = _LP_STEPS + [_step_lp(2, -0.01, [(2, -0.01)])]
        _, client = _make_lp_server([100, 101, 102, 2], steps, _LP_VOCAB)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "logprobs": 1,
        })
        lp = resp.json()["choices"][0]["logprobs"]
        assert lp["tokens"] == ["Hello", " world", "!"]
        assert lp["token_logprobs"] == [-0.1, -0.2, -0.3]

    def test_logprobs_21_rejected(self):
        _, client = _make_lp_server([100], None, _LP_VOCAB)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "logprobs": 21,
        })
        assert resp.status_code == 422


class TestChatLogprobs:

    def _post(self, client, **overrides):
        body = {"model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}]}
        body.update(overrides)
        return client.post("/v1/chat/completions", json=body)

    def test_content_block_shape(self):
        _, client = _make_lp_server([100, 101, 102], _LP_STEPS, _LP_VOCAB)
        resp = self._post(client, logprobs=True, top_logprobs=2)
        assert resp.status_code == 200
        lp = resp.json()["choices"][0]["logprobs"]
        assert lp is not None and len(lp["content"]) == 3
        e0 = lp["content"][0]
        assert e0["token"] == "Hello"
        assert e0["logprob"] == -0.1
        assert e0["bytes"] == list(b"Hello")
        assert [t["token"] for t in e0["top_logprobs"]] == ["Hello", "alt"]

    def test_logprobs_true_without_top_maps_to_k0(self):
        steps = [_step_lp(100, -0.1, []), _step_lp(101, -0.2, [])]
        server, client = _make_lp_server([100, 101], steps, _LP_VOCAB)
        resp = self._post(client, logprobs=True)
        inf_req = server._orchestrator.submit_request.call_args[0][0]
        assert inf_req.logprobs == 0          # chosen-token only
        lp = resp.json()["choices"][0]["logprobs"]
        assert len(lp["content"]) == 2
        assert lp["content"][0]["top_logprobs"] == []

    def test_logprobs_absent_serves_null(self):
        server, client = _make_lp_server([100, 101], None, _LP_VOCAB)
        resp = self._post(client)
        assert resp.json()["choices"][0]["logprobs"] is None
        inf_req = server._orchestrator.submit_request.call_args[0][0]
        assert inf_req.logprobs is None

    def test_top_logprobs_21_rejected(self):
        _, client = _make_lp_server([100], None, _LP_VOCAB)
        resp = self._post(client, logprobs=True, top_logprobs=21)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Orchestrator failures → OpenAI HTTP errors (TD-SERVE-ERROR-MASKING).
# A request the orchestrator FAILS must never come back as 200 + empty
# body: non-streaming answers with the OpenAI error object and a real
# status code, streaming terminates the event stream with the same object
# instead of a fabricated finish_reason chunk.
# ---------------------------------------------------------------------------

# Real engine wording (src/daemon/dispatch_lifecycle.cpp): transient
# exhaustion (the pool COULD hold the prompt) vs a prompt that does not
# fit an empty pool.
_POOL_BUSY = ("seq_create: kMain pool exhausted (need 480 pages = 8 logical "
              "x 60 layers, got 12, pool 12/4096 free gpu0)")
_POOL_TOO_BIG = ("seq_create: kMain pool exhausted (need 8192 pages = 128 "
                 "logical x 64 layers, got 0, pool 4096/4096 free gpu0)")


def _failing_orchestrator(
    detail: str,
    *,
    tokens: list[int] | None = None,
    reason: str = "error",
    legacy: bool = False,
):
    """Mock orchestrator that streams ``tokens`` (if any) and then reports
    ``reason``.  ``legacy=True`` uses the four-argument on_complete
    contract (no error detail) — the pre-TD consumers must keep working."""
    orch = MagicMock()

    def fake_submit(req: InferenceRequest) -> None:
        for tok in tokens or []:
            if req.on_token is not None:
                req.on_token(req.request_id, tok, None)
        if req.on_complete is None:
            return
        if legacy:
            req.on_complete(req.request_id, tokens or [], reason, None)
        else:
            req.on_complete(req.request_id, tokens or [], reason, None,
                            error=detail)

    orch.submit_request.side_effect = fake_submit
    return orch


def _failing_server(detail: str, **kw) -> TestClient:
    server = LayerStoRmServer(
        orchestrator=_failing_orchestrator(detail, **kw),
        tokenizer=_mock_tokenizer("Hello world"),
        chat_template=_mock_chat_template(),
        metadata=_metadata((2,)),
        model_name="test-model",
    )
    return TestClient(server.app)


def _sse_payloads(text: str) -> list[str]:
    return [ln[6:] for ln in text.split("\n") if ln.startswith("data: ")]


def _post_completion(client: TestClient, **extra):
    body = {"model": "test-model", "prompt": "Hello"}
    body.update(extra)
    return client.post("/v1/completions", json=body)


def _post_chat(client: TestClient, **extra):
    body = {"model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}]}
    body.update(extra)
    return client.post("/v1/chat/completions", json=body)


class TestClassifyRequestError:
    """Status-code mapping table for the named engine failures."""

    @pytest.mark.parametrize("detail,expected", [
        (_POOL_BUSY, (503, "server_error", "kv_cache_exhausted")),
        ("prefix-cache fork: pool exhausted after evicting every entry",
         (503, "server_error", "kv_cache_exhausted")),
        (_POOL_TOO_BIG, (400, "invalid_request_error",
                         "context_length_exceeded")),
        ("empty prompt", (400, "invalid_request_error", "invalid_prompt")),
        ("prompt token out of vocab: 999999",
         (400, "invalid_request_error", "invalid_prompt")),
        ("timeout waiting for cmp 0x3 (decode)",
         (504, "server_error", "engine_timeout")),
        ("CMP_ERROR (far forward): kernel launch failed",
         (500, "server_error", "internal_error")),
        ("", (500, "server_error", "internal_error")),
    ])
    def test_mapping(self, detail, expected):
        from server.http_server import classify_request_error
        assert classify_request_error(detail) == expected

    def test_error_body_shape_and_retry_after(self):
        from server.http_server import request_error_parts
        status, body, headers = request_error_parts(_POOL_BUSY)
        assert status == 503
        assert headers == {"Retry-After": "1"}
        dumped = body.model_dump()
        assert set(dumped) == {"error"}
        assert set(dumped["error"]) == {"message", "type", "param", "code"}
        assert dumped["error"]["message"] == _POOL_BUSY

    def test_sse_event_matches_json_body(self):
        from server.http_server import request_error_parts, sse_error_event
        _, body, _ = request_error_parts(_POOL_BUSY)
        ev = sse_error_event(_POOL_BUSY)
        assert ev.startswith("data: ") and ev.endswith("\n\n")
        assert json.loads(ev[6:]) == body.model_dump()


class TestRequestErrorNonStreaming:

    def test_completion_internal_error_500(self):
        client = _failing_server("CMP_ERROR (decode): boom")
        resp = _post_completion(client)
        assert resp.status_code == 500
        err = resp.json()["error"]
        assert err["type"] == "server_error"
        assert err["code"] == "internal_error"
        assert "boom" in err["message"]

    def test_chat_internal_error_500(self):
        client = _failing_server("CMP_ERROR (decode): boom")
        resp = _post_chat(client)
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "internal_error"

    def test_completion_pool_exhausted_503(self):
        client = _failing_server(_POOL_BUSY)
        resp = _post_completion(client)
        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "1"
        err = resp.json()["error"]
        assert err["code"] == "kv_cache_exhausted"
        assert "pool exhausted" in err["message"]

    def test_chat_pool_exhausted_503(self):
        client = _failing_server(_POOL_BUSY)
        resp = _post_chat(client)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "kv_cache_exhausted"

    def test_prompt_larger_than_pool_is_400(self):
        client = _failing_server(_POOL_TOO_BIG)
        resp = _post_completion(client)
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "context_length_exceeded"

    def test_bad_prompt_is_400(self):
        client = _failing_server("prompt token out of vocab: 999999")
        resp = _post_completion(client, prompt=[1, 2, 999999])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_prompt"

    def test_engine_timeout_is_504(self):
        client = _failing_server("timeout waiting for cmp 0x3 (decode)")
        resp = _post_chat(client)
        assert resp.status_code == 504
        assert resp.json()["error"]["code"] == "engine_timeout"

    def test_legacy_four_arg_callback_still_surfaces_500(self):
        """A consumer chain that never delivers a detail must still fail
        the request (generic 500), not return an empty 200."""
        client = _failing_server("", legacy=True)
        resp = _post_completion(client)
        assert resp.status_code == 500
        assert resp.json()["error"]["message"] == "internal engine error"

    def test_error_body_carries_no_choices(self):
        client = _failing_server("CMP_ERROR: boom")
        body = _post_chat(client).json()
        assert "choices" not in body and "usage" not in body

    def test_cancelled_is_not_an_error(self):
        client = _failing_server("", reason="cancelled", tokens=[100])
        resp = _post_completion(client)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "cancelled"

    def test_success_path_unchanged(self):
        client = _failing_server("", reason="stop", tokens=[100, 101])
        resp = _post_completion(client)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "Hello world"


class TestRequestErrorStreaming:

    def _events(self, resp) -> list[str]:
        assert resp.status_code == 200          # headers precede the failure
        assert resp.headers["content-type"].startswith("text/event-stream")
        return _sse_payloads(resp.text)

    def test_completion_error_before_first_token(self):
        client = _failing_server(_POOL_BUSY)
        events = self._events(_post_completion(client, stream=True))
        assert len(events) == 2
        err = json.loads(events[0])["error"]
        assert err["code"] == "kv_cache_exhausted"
        assert err["type"] == "server_error"
        assert events[-1] == "[DONE]"

    def test_completion_error_mid_stream(self):
        client = _failing_server("CMP_ERROR (decode): boom",
                                 tokens=[100, 101])
        events = self._events(_post_completion(client, stream=True))
        assert events[-1] == "[DONE]"
        payloads = [json.loads(e) for e in events[:-1]]
        # deltas were already sent, then the stream ends on the error —
        # never on a fabricated finish_reason.
        assert any(p.get("choices") for p in payloads)
        assert not any(c.get("finish_reason")
                       for p in payloads if "choices" in p
                       for c in p["choices"])
        assert payloads[-1]["error"]["code"] == "internal_error"

    def test_chat_error_before_first_token(self):
        client = _failing_server(_POOL_TOO_BIG)
        events = self._events(_post_chat(client, stream=True))
        assert len(events) == 2
        err = json.loads(events[0])["error"]
        assert err["code"] == "context_length_exceeded"
        assert err["type"] == "invalid_request_error"
        assert events[-1] == "[DONE]"

    def test_chat_error_mid_stream(self):
        client = _failing_server("timeout waiting for cmp 0x3 (decode)",
                                 tokens=[100, 101])
        events = self._events(_post_chat(client, stream=True))
        assert events[-1] == "[DONE]"
        payloads = [json.loads(e) for e in events[:-1]]
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert not any(c.get("finish_reason")
                       for p in payloads if "choices" in p
                       for c in p["choices"])
        assert payloads[-1]["error"]["code"] == "engine_timeout"

    def test_chat_cancelled_stream_unchanged(self):
        client = _failing_server("", reason="cancelled", tokens=[100])
        events = self._events(_post_chat(client, stream=True))
        payloads = [json.loads(e) for e in events[:-1]]
        assert not any("error" in p for p in payloads)
        assert payloads[-1]["choices"][0]["finish_reason"] == "cancelled"

    def test_completion_success_stream_unchanged(self):
        client = _failing_server("", reason="stop", tokens=[100, 101])
        events = self._events(_post_completion(client, stream=True))
        payloads = [json.loads(e) for e in events[:-1]]
        assert not any("error" in p for p in payloads)
        assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
        assert payloads[-1]["usage"]["completion_tokens"] == 2
