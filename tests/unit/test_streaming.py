"""Tests for SSE streaming responses."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from orchestrator.loop.orchestrator_loop import InferenceRequest
from orchestrator.types import EngineMetadata, GpuConfig
from server.http_server import LayerStoRmServer
from server.streaming import (
    ChatCompletionStreamResponse,
    CompletionStreamResponse,
    TokenQueue,
    _sse_done,
    _sse_line,
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


def _mock_tokenizer(vocab: dict[int, str] | None = None):
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]

    if vocab is None:
        vocab = {100: "Hello", 101: " world", 102: "!"}

    def decode_fn(ids):
        return "".join(vocab.get(t, "") for t in ids)

    tok.decode.side_effect = decode_fn
    return tok


def _mock_chat_template():
    tmpl = MagicMock()
    tmpl.render.return_value = "<|begin|>Hello<|end|>"
    return tmpl


def _streaming_orchestrator(
    tokens: list[int],
    finish_reason: str = "stop",
    delay: float = 0.0,
    lps: list | None = None,
):
    orch = MagicMock()

    def fake_submit(req: InferenceRequest) -> None:
        def emit():
            for i, tok in enumerate(tokens):
                if delay > 0:
                    time.sleep(delay)
                if req.on_token is not None:
                    req.on_token(req.request_id, tok,
                                 lps[i] if lps is not None else None)
            if req.on_complete is not None:
                req.on_complete(req.request_id, tokens, finish_reason, lps)

        t = threading.Thread(target=emit, daemon=True)
        t.start()

    orch.submit_request.side_effect = fake_submit
    return orch


def _make_streaming_server(
    *,
    tokens: list[int] | None = None,
    finish_reason: str = "stop",
    eos: tuple[int, ...] = (2,),
    model_type: str = "",
    vocab: dict[int, str] | None = None,
    lps: list | None = None,
) -> tuple[LayerStoRmServer, TestClient]:
    if tokens is None:
        tokens = [100, 101, 102]
    server = LayerStoRmServer(
        orchestrator=_streaming_orchestrator(tokens, finish_reason,
                                             lps=lps),
        tokenizer=_mock_tokenizer(vocab),
        chat_template=_mock_chat_template(),
        metadata=_metadata(eos),
        model_name="test-model",
        model_type=model_type,
    )
    client = TestClient(server.app)
    return server, client


def _parse_sse_events(content: str) -> list[str]:
    """Parse SSE content into individual data payloads."""
    events = []
    for line in content.split("\n"):
        if line.startswith("data: "):
            events.append(line[6:])
    return events


# ---------------------------------------------------------------------------
# TokenQueue
# ---------------------------------------------------------------------------

class TestTokenQueue:

    def test_push_and_drain(self):
        tq = TokenQueue()
        tq.push(1, 100)
        tq.push(1, 101)
        items = tq.drain()
        assert [t for t, _ in items] == [100, 101]
        assert tq.drain() == []

    def test_mark_done(self):
        tq = TokenQueue()
        assert not tq.done
        tq.mark_done(1, [100], "stop")
        assert tq.done
        assert tq.finish_reason == "stop"

    def test_finish_reason_default(self):
        tq = TokenQueue()
        assert tq.finish_reason == "stop"

    def test_mark_done_carries_error_detail(self):
        """TD-SERVE-ERROR-MASKING: the failure detail reaches the SSE
        generator; four-argument callers keep working and stay unfailed."""
        tq = TokenQueue()
        assert not tq.failed and tq.error == ""
        tq.mark_done(1, [], "error", None, "seq_create: kMain pool exhausted")
        assert tq.done and tq.failed
        assert tq.error == "seq_create: kMain pool exhausted"

        legacy = TokenQueue()
        legacy.mark_done(1, [100], "stop")
        assert legacy.done and not legacy.failed and legacy.error == ""

    def test_thread_safety(self):
        tq = TokenQueue()
        count = 1000

        def producer():
            for i in range(count):
                tq.push(1, i)

        t = threading.Thread(target=producer)
        t.start()
        t.join()

        all_tokens = tq.drain()
        assert len(all_tokens) == count


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

class TestSseHelpers:

    def test_sse_line_format(self):
        result = _sse_line('{"key":"value"}')
        assert result == 'data: {"key":"value"}\n\n'

    def test_sse_done(self):
        assert _sse_done() == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Completion streaming
# ---------------------------------------------------------------------------

class TestCompletionStreaming:

    def test_stream_basic(self):
        _, client = _make_streaming_server(tokens=[100, 101, 102])
        with client.stream(
            "POST", "/v1/completions",
            json={"model": "test-model", "prompt": "Hello", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
            content = resp.read().decode()

        events = _parse_sse_events(content)
        assert events[-1] == "[DONE]"

        data_events = [json.loads(e) for e in events if e != "[DONE]"]
        assert len(data_events) >= 2

        for ev in data_events:
            assert ev["object"] == "text_completion"
            assert ev["id"].startswith("cmpl-")
            assert ev["model"] == "test-model"

    def test_stream_delta_accumulation(self):
        _, client = _make_streaming_server(tokens=[100, 101, 102])
        with client.stream(
            "POST", "/v1/completions",
            json={"model": "test-model", "prompt": "Hi", "stream": True},
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        full_text = "".join(
            ev["choices"][0]["text"] for ev in data_events
        )
        assert full_text == "Hello world!"

    def test_stream_finish_reason_in_final_chunk(self):
        _, client = _make_streaming_server(
            tokens=[100, 101], finish_reason="length",
        )
        with client.stream(
            "POST", "/v1/completions",
            json={"model": "test-model", "prompt": "Hi", "stream": True},
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        non_final = data_events[:-1]
        for ev in non_final:
            assert ev["choices"][0].get("finish_reason") is None

        final = data_events[-1]
        assert final["choices"][0]["finish_reason"] == "length"

    def test_stream_usage_in_final_chunk(self):
        _, client = _make_streaming_server(tokens=[100, 101])
        with client.stream(
            "POST", "/v1/completions",
            json={"model": "test-model", "prompt": "Hi", "stream": True},
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        final = data_events[-1]
        assert "usage" in final
        assert final["usage"]["prompt_tokens"] == 3
        assert final["usage"]["completion_tokens"] == 2
        assert final["usage"]["total_tokens"] == 5

    def test_stream_eos_stripped(self):
        eos_id = 2
        _, client = _make_streaming_server(
            tokens=[100, 101, eos_id],
            eos=(eos_id,),
        )
        with client.stream(
            "POST", "/v1/completions",
            json={"model": "test-model", "prompt": "Hi", "stream": True},
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        full_text = "".join(
            ev["choices"][0]["text"] for ev in data_events
        )
        assert full_text == "Hello world"

    def test_stream_stop_sequence(self):
        vocab = {100: "Hello", 101: " STOP", 102: " more"}
        _, client = _make_streaming_server(
            tokens=[100, 101, 102], vocab=vocab,
        )
        with client.stream(
            "POST", "/v1/completions",
            json={
                "model": "test-model", "prompt": "Hi",
                "stream": True, "stop": ["STOP"],
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        full_text = "".join(
            ev["choices"][0]["text"] for ev in data_events
        )
        assert "STOP" not in full_text
        assert full_text == "Hello "


# ---------------------------------------------------------------------------
# Chat completion streaming
# ---------------------------------------------------------------------------

class TestChatCompletionStreaming:

    def test_stream_basic(self):
        _, client = _make_streaming_server(tokens=[100, 101])
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            content = resp.read().decode()

        events = _parse_sse_events(content)
        assert events[-1] == "[DONE]"

        data_events = [json.loads(e) for e in events if e != "[DONE]"]
        assert len(data_events) >= 2

        for ev in data_events:
            assert ev["object"] == "chat.completion.chunk"
            assert ev["id"].startswith("chatcmpl-")

    def test_stream_first_chunk_has_role(self):
        _, client = _make_streaming_server(tokens=[100, 101])
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        first = data_events[0]
        assert first["choices"][0]["delta"]["role"] == "assistant"

    def test_stream_content_accumulation(self):
        _, client = _make_streaming_server(tokens=[100, 101, 102])
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        full_content = ""
        for ev in data_events:
            delta = ev["choices"][0]["delta"]
            if delta.get("content"):
                full_content += delta["content"]
        assert full_content == "Hello world!"

    def test_stream_finish_reason_in_final(self):
        _, client = _make_streaming_server(
            tokens=[100], finish_reason="stop",
        )
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        final = data_events[-1]
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_stream_tool_calls_in_final(self):
        tool_vocab = {
            200: "<｜tool▁calls▁begin｜>",
            201: "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                 "```json\n"
                 '{"city": "NYC"}\n'
                 "```\n"
                 "<｜tool▁call▁end｜>",
            202: "<｜tool▁calls▁end｜>",
        }
        _, client = _make_streaming_server(
            tokens=[200, 201, 202],
            model_type="deepseek_v32",
            vocab=tool_vocab,
        )
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": True,
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        final = data_events[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        tc = final["choices"][0]["delta"].get("tool_calls")
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_stream_usage_in_final(self):
        _, client = _make_streaming_server(tokens=[100])
        with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            content = resp.read().decode()

        events = _parse_sse_events(content)
        data_events = [json.loads(e) for e in events if e != "[DONE]"]

        final = data_events[-1]
        assert final["usage"]["prompt_tokens"] == 3
        assert final["usage"]["completion_tokens"] == 1


# ---------------------------------------------------------------------------
# Non-streaming still works
# ---------------------------------------------------------------------------

class TestNonStreamingUnchanged:

    def test_completions_non_stream(self):
        orch = MagicMock()

        def fake_submit(req):
            if req.on_complete:
                req.on_complete(req.request_id, [100, 101], "stop", None)

        orch.submit_request.side_effect = fake_submit
        server = LayerStoRmServer(
            orchestrator=orch,
            tokenizer=_mock_tokenizer(),
            chat_template=_mock_chat_template(),
            metadata=_metadata(),
            model_name="test-model",
        )
        client = TestClient(server.app)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "Hello world"

    def test_chat_completions_non_stream(self):
        orch = MagicMock()

        def fake_submit(req):
            if req.on_complete:
                req.on_complete(req.request_id, [100], "stop", None)

        orch.submit_request.side_effect = fake_submit
        server = LayerStoRmServer(
            orchestrator=orch,
            tokenizer=_mock_tokenizer(),
            chat_template=_mock_chat_template(),
            metadata=_metadata(),
            model_name="test-model",
        )
        client = TestClient(server.app)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# Named parsers: glm45 reasoning deltas + glm47 incremental tool deltas
# ---------------------------------------------------------------------------

THINK_START = 90
THINK_END = 91

_PARSER_VOCAB = {
    90: "<think>",
    91: "</think>",
    10: "Weather check",
    20: "<tool_call>",
    21: "get_weather",
    22: "<arg_key>",
    23: "city",
    24: "</arg_key>",
    25: "<arg_value>",
    26: "Par",
    29: "is",
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

_TOOL_TOKENS = [10, 91, 20, 21, 22, 23, 24, 25, 26, 29, 27, 28, 2]
_REASONING_TOKENS = [10, 91, 30, 2]


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


def _make_parser_streaming_server(
    tokens: list[int],
    *,
    delay: float = 0.003,
) -> tuple[LayerStoRmServer, TestClient]:
    server = LayerStoRmServer(
        orchestrator=_streaming_orchestrator(tokens, "stop", delay),
        tokenizer=_mock_tokenizer(_PARSER_VOCAB),
        chat_template=_mock_chat_template(),
        metadata=_metadata_with_think(),
        model_name="test-model",
        tool_call_parser="glm47",
        enable_auto_tool_choice=True,
        reasoning_parser="glm45",
    )
    return server, TestClient(server.app)


def _stream_chat(client, **overrides):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "stream": True,
        "tools": _WEATHER_TOOLS,
    }
    body.update(overrides)
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


class TestParserStreaming:

    def test_reasoning_then_tool_call_deltas(self):
        _, client = _make_parser_streaming_server(_TOOL_TOKENS)
        chunks = _stream_chat(client)
        deltas = [c["choices"][0]["delta"] for c in chunks]

        # role priming chunk first
        assert deltas[0].get("role") == "assistant"

        reasoning = "".join(
            d.get("reasoning_content") or "" for d in deltas)
        content = "".join(d.get("content") or "" for d in deltas)
        assert reasoning == "Weather check"
        assert content == ""

        tc_deltas = [tc for d in deltas for tc in (d.get("tool_calls") or [])]
        assert tc_deltas, "expected incremental tool_call deltas"
        # First tool delta announces index/id/type/name.
        first = tc_deltas[0]
        assert first["index"] == 0
        assert first["id"] == "call_0"
        assert first["type"] == "function"
        assert first["function"]["name"] == "get_weather"
        args = "".join(
            tc["function"].get("arguments") or "" for tc in tc_deltas)
        assert json.loads(args) == {"city": "Paris"}

        # reasoning deltas must precede any tool-call delta
        first_tool_at = next(
            i for i, d in enumerate(deltas) if d.get("tool_calls"))
        last_reasoning_at = max(
            i for i, d in enumerate(deltas) if d.get("reasoning_content"))
        assert last_reasoning_at < first_tool_at

        final = chunks[-1]["choices"][0]
        assert final["finish_reason"] == "tool_calls"
        assert chunks[-1]["usage"]["completion_tokens"] == len(_TOOL_TOKENS) - 1

    def test_incremental_arguments_across_chunks(self):
        # Per-token delay forces one drain per token: the string-typed
        # argument value must arrive in MULTIPLE fragments.
        _, client = _make_parser_streaming_server(_TOOL_TOKENS, delay=0.02)
        chunks = _stream_chat(client)
        deltas = [c["choices"][0]["delta"] for c in chunks]
        arg_frags = [
            tc["function"]["arguments"]
            for d in deltas for tc in (d.get("tool_calls") or [])
            if tc["function"].get("arguments")
        ]
        assert len(arg_frags) >= 3
        assert any("Par" in f and "is" not in f for f in arg_frags)

    def test_reasoning_content_separation(self):
        _, client = _make_parser_streaming_server(_REASONING_TOKENS)
        chunks = _stream_chat(client, tools=None)
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
        content = "".join(d.get("content") or "" for d in deltas)
        assert reasoning == "Weather check"
        assert content == "The answer."
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        # no delta mixes both channels
        assert not any(
            d.get("reasoning_content") and d.get("content") for d in deltas)

    def test_thinking_false_all_content(self):
        _, client = _make_parser_streaming_server([30, 2])
        chunks = _stream_chat(client, tools=None, thinking=False)
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert "".join(d.get("content") or "" for d in deltas) == \
            "The answer."
        assert not any(d.get("reasoning_content") for d in deltas)

    def test_tool_choice_none_streams_raw_text(self):
        _, client = _make_parser_streaming_server(_TOOL_TOKENS)
        chunks = _stream_chat(client, tool_choice="none")
        deltas = [c["choices"][0]["delta"] for c in chunks]
        content = "".join(d.get("content") or "" for d in deltas)
        assert "<tool_call>" in content
        assert not any(d.get("tool_calls") for d in deltas)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Guided decoding streaming (TD-SERVE-NAMED-TOOL-CHOICE): named tool_choice
# streams a valid delta sequence (open delta with id/name, argument
# fragments concatenating to valid JSON, finish_reason tool_calls).
# ---------------------------------------------------------------------------


class _StubGuidedManagerStreaming:
    def build(self, tools, tool_choice, reasoning, model="glm_4_7"):
        class _G:
            def new_state(self):
                return object()
        return _G()


class TestGuidedStreaming:

    def test_named_tool_choice_streams_tool_call_deltas(self):
        server = LayerStoRmServer(
            orchestrator=_streaming_orchestrator(_TOOL_TOKENS, "tool_calls",
                                                 0.003),
            tokenizer=_mock_tokenizer(_PARSER_VOCAB),
            chat_template=_mock_chat_template(),
            metadata=_metadata_with_think(),
            model_name="test-model",
            tool_call_parser="glm47",
            enable_auto_tool_choice=True,
            reasoning_parser="glm45",
            guided_manager=_StubGuidedManagerStreaming(),
        )
        client = TestClient(server.app)
        chunks = _stream_chat(client, tool_choice={
            "type": "function", "function": {"name": "get_weather"}})
        deltas = [c["choices"][0]["delta"] for c in chunks]
        tc_deltas = [tc for d in deltas
                     for tc in (d.get("tool_calls") or [])]
        assert tc_deltas, "no tool-call deltas under named tool_choice"
        opener = tc_deltas[0]
        assert opener["id"] == "call_0" and opener["type"] == "function"
        assert opener["function"]["name"] == "get_weather"
        args = "".join(tc["function"].get("arguments", "")
                       for tc in tc_deltas)
        assert json.loads(args) == {"city": "Paris"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        req = server._orchestrator.submit_request.call_args[0][0]
        assert req.guided is not None


# ---------------------------------------------------------------------------
# SSE logprobs (TD-ORCH-LOGPROBS): per-chunk logprobs blocks
# ---------------------------------------------------------------------------

def _mk_lp(tid: int, lp: float, tops: list[tuple[int, float]]):
    from orchestrator.types import StepLogprobs, TokenLogprob
    return StepLogprobs(token=TokenLogprob(tid, lp),
                        top_logprobs=tuple(TokenLogprob(t, v)
                                           for t, v in tops))


_STREAM_LPS = [
    _mk_lp(100, -0.1, [(100, -0.1)]),
    _mk_lp(101, -0.2, [(101, -0.2)]),
    _mk_lp(102, -0.3, [(102, -0.3)]),
]


class TestStreamingLogprobs:

    def test_completion_stream_carries_logprobs(self):
        _, client = _make_streaming_server(lps=_STREAM_LPS)
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "stream": True,
            "logprobs": 1,
        })
        events = _parse_sse_events(resp.text)
        chunks = [json.loads(e) for e in events if e != "[DONE]"]
        toks, lp_vals, offsets = [], [], []
        for c in chunks:
            lp = c["choices"][0].get("logprobs")
            if lp:
                toks += lp["tokens"]
                lp_vals += lp["token_logprobs"]
                offsets += lp["text_offset"]
        assert toks == ["Hello", " world", "!"]
        assert lp_vals == [-0.1, -0.2, -0.3]
        # Absolute offsets survive chunking.
        assert offsets == [0, 5, 11]

    def test_completion_stream_no_logprobs_field_when_off(self):
        _, client = _make_streaming_server()
        resp = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "stream": True,
        })
        for e in _parse_sse_events(resp.text):
            if e == "[DONE]":
                continue
            for choice in json.loads(e)["choices"]:
                assert "logprobs" not in choice     # exclude_none intact

    def test_chat_stream_carries_logprobs_content(self):
        _, client = _make_streaming_server(lps=_STREAM_LPS)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
            "logprobs": True, "top_logprobs": 1,
        })
        events = _parse_sse_events(resp.text)
        entries = []
        for e in events:
            if e == "[DONE]":
                continue
            lp = json.loads(e)["choices"][0].get("logprobs")
            if lp and lp.get("content"):
                entries += lp["content"]
        assert [t["token"] for t in entries] == ["Hello", " world", "!"]
        assert [t["logprob"] for t in entries] == [-0.1, -0.2, -0.3]
        assert all(len(t["top_logprobs"]) == 1 for t in entries)

    def test_chat_stream_no_logprobs_field_when_off(self):
        _, client = _make_streaming_server()
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        for e in _parse_sse_events(resp.text):
            if e == "[DONE]":
                continue
            for choice in json.loads(e)["choices"]:
                assert "logprobs" not in choice


# ---------------------------------------------------------------------------
# Teardown contract (INV-SERVE-CANCEL, 2026-08-24 zombie-generation fix):
# ANY generator exit while the orchestrator is still generating must call
# cancel_fn — including GeneratorExit/aclose (how starlette actually tears
# down an SSE response after a client disconnect: the generator is never
# resumed, so a poll-only cancel never fires).  release_fn (admission slot)
# runs on every exit.
# ---------------------------------------------------------------------------

class _FakeRequest:
    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _cancel_recorder():
    calls: list[int] = []
    return calls, calls.append


class TestStreamTeardownCancels:

    def _gen(self, tq, raw, cancel_fn, release_fn, *, chat=False,
             stop=None):
        from server.streaming import (stream_chat_completion_response,
                                      stream_completion_response)
        kw = dict(
            request_id=7,
            response_id="cmpl-7",
            model="m",
            token_queue=tq,
            tokenizer=_mock_tokenizer(),
            eos_token_ids=(2,),
            raw_request=raw,
            cancel_fn=cancel_fn,
            stop=stop,
            prompt_tokens=3,
            release_fn=release_fn,
        )
        if chat:
            return stream_chat_completion_response(
                tool_parser=None, **kw)
        return stream_completion_response(**kw)

    def test_aclose_mid_stream_cancels_and_releases(self):
        """GeneratorExit (the real starlette disconnect path) cancels the
        still-running generation and frees the slot."""
        import asyncio

        async def scenario(chat: bool):
            tq = TokenQueue()
            tq.push(7, 100)                      # generation in flight
            cancels, cancel_fn = _cancel_recorder()
            releases, release_fn = _cancel_recorder()
            gen = self._gen(tq, _FakeRequest(), cancel_fn,
                            lambda: release_fn(0), chat=chat)
            await gen.__anext__()                # stream one chunk
            await gen.aclose()                   # ASGI teardown
            assert cancels == [7]
            assert len(releases) == 1

        asyncio.run(scenario(chat=False))
        asyncio.run(scenario(chat=True))

    def test_disconnect_poll_cancels_and_releases(self):
        """The in-loop disconnect poll still works (idle stream, no chunk
        in flight when the client vanishes)."""
        import asyncio

        async def scenario(chat: bool):
            tq = TokenQueue()                    # no tokens, not done
            cancels, cancel_fn = _cancel_recorder()
            releases, release_fn = _cancel_recorder()
            gen = self._gen(tq, _FakeRequest(disconnected=True), cancel_fn,
                            lambda: release_fn(0), chat=chat)
            chunks = [c async for c in gen]      # returns immediately
            assert chunks == []
            assert cancels == [7]
            assert len(releases) == 1

        asyncio.run(scenario(chat=False))
        asyncio.run(scenario(chat=True))

    def test_normal_completion_releases_without_cancel(self):
        import asyncio

        async def scenario(chat: bool):
            tq = TokenQueue()
            tq.push(7, 100)
            tq.push(7, 101)
            tq.mark_done(7, [100, 101], "stop")
            cancels, cancel_fn = _cancel_recorder()
            releases, release_fn = _cancel_recorder()
            gen = self._gen(tq, _FakeRequest(), cancel_fn,
                            lambda: release_fn(0), chat=chat)
            chunks = [c async for c in gen]
            assert chunks[-1] == "data: [DONE]\n\n"
            assert cancels == []                 # done — nothing to cancel
            assert len(releases) == 1

        asyncio.run(scenario(chat=False))
        asyncio.run(scenario(chat=True))

    def test_stop_sequence_truncation_cancels_generation(self):
        """Server-side stop truncation ends the RESPONSE — the engine
        generation is still running and must be cancelled too."""
        import asyncio

        async def scenario(chat: bool):
            tq = TokenQueue()
            tq.push(7, 100)
            tq.push(7, 101)                      # "Hello world" ⊃ "wor"
            cancels, cancel_fn = _cancel_recorder()
            releases, release_fn = _cancel_recorder()
            gen = self._gen(tq, _FakeRequest(), cancel_fn,
                            lambda: release_fn(0), chat=chat, stop="wor")
            chunks = [c async for c in gen]
            assert chunks[-1] == "data: [DONE]\n\n"
            assert cancels == [7]                # queue not done → cancel
            assert len(releases) == 1

        asyncio.run(scenario(chat=False))
        asyncio.run(scenario(chat=True))
