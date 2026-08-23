"""OpenAI-compatible HTTP server for LayerStoRm inference.

FastAPI/uvicorn server exposing ``/v1/completions``, ``/v1/chat/completions``,
``/v1/models``, and ``/health``.  Bridges async HTTP handlers with the
synchronous single-threaded orchestrator via a thread-safe ``_RequestTracker``.

API-surface parity details (streamed-error SSE shape, tokenizer-mode
resolution, reasoning/tool-choice semantics) follow vLLM (ref/vllm;
Apache-2.0, Copyright contributors to the vLLM project — see
THIRD_PARTY_NOTICES.md).

Streaming responses are deferred to #68 (SSE).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator

from orchestrator.orchestrator import (InferenceRequest, Orchestrator,
                                       SamplingParams)
from orchestrator.types import EngineMetadata
from server.guided import GuidedDecodingManager, GuidedError
from server.reasoning_parsers import get_reasoning_parser
from server.tool_parsers import get_tool_parser
from tokenizer import ChatTemplateRenderer, TokenizerWrapper, get_tool_call_parser

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[int]
    max_tokens: int | None = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    logprobs: int | None = None
    user: str | None = None

    @field_validator("n")
    @classmethod
    def _n_must_be_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n > 1 is not supported")
        return v

    @field_validator("logprobs")
    @classmethod
    def _logprobs_range(cls, v: int | None) -> int | None:
        if v is not None and (v < 0 or v > 20):
            raise ValueError("logprobs must be between 0 and 20")
        return v


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    thinking: bool | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None
    user: str | None = None

    @field_validator("n")
    @classmethod
    def _n_must_be_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n > 1 is not supported")
        return v

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    @field_validator("top_logprobs")
    @classmethod
    def _top_logprobs_range(cls, v: int | None) -> int | None:
        if v is not None and (v < 0 or v > 20):
            raise ValueError("top_logprobs must be between 0 and 20")
        return v

    @field_validator("tool_choice")
    @classmethod
    def _tool_choice_shape(
        cls, v: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        """OpenAI shapes only: "auto" | "none" | "required" | named
        function object.  Semantic support (e.g. named forcing) is checked
        by the server; this validator only rejects malformed values."""
        if v is None or v in ("auto", "none", "required"):
            return v
        if isinstance(v, dict):
            fn = v.get("function")
            if (v.get("type") == "function" and isinstance(fn, dict)
                    and isinstance(fn.get("name"), str) and fn["name"]):
                return v
            raise ValueError(
                'named tool_choice must be {"type": "function", '
                '"function": {"name": ...}}')
        raise ValueError(
            'tool_choice must be "auto", "none", "required" or a named '
            "function object")


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionLogprobs(BaseModel):
    text_offset: list[int]
    token_logprobs: list[float | None]
    tokens: list[str]
    top_logprobs: list[dict[str, float] | None]


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None
    logprobs: CompletionLogprobs | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ToolCallFunctionResponse(BaseModel):
    name: str
    arguments: str


class ToolCallResponse(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunctionResponse


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallResponse] | None = None


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class ChatCompletionTokenLogprobContent(ChatCompletionTokenLogprob):
    top_logprobs: list[ChatCompletionTokenLogprob]


class ChatCompletionLogprobs(BaseModel):
    content: list[ChatCompletionTokenLogprobContent] | None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "layerstorm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Orchestrator failure → HTTP error mapping (TD-SERVE-ERROR-MASKING)
# ---------------------------------------------------------------------------
#
# A request that dies inside the orchestrator finishes with reason "error"
# and a detail string (InferenceRequest.on_complete's optional ``error``
# arg).  It MUST surface as an OpenAI-shaped HTTP error — never as a 200
# carrying an empty completion (INV-SERVE-ERROR).  The named engine
# failures are classified here; everything unrecognised is a 500.

# "seq_create: kMain pool exhausted (need 480 pages = 8 logical x 60
#  layers, got 12, pool 12/4096 free gpu0)" — src/daemon/dispatch_lifecycle.cpp
_POOL_NEED_RE = re.compile(r"need (\d+) pages")
_POOL_TOTAL_RE = re.compile(r"pool \d+/(\d+) free")

_ERR_INTERNAL = (500, "server_error", "internal_error")
_ERR_CAPACITY = (503, "server_error", "kv_cache_exhausted")
_ERR_TOO_LONG = (400, "invalid_request_error", "context_length_exceeded")
_ERR_BAD_PROMPT = (400, "invalid_request_error", "invalid_prompt")
_ERR_TIMEOUT = (504, "server_error", "engine_timeout")


def classify_request_error(detail: str) -> tuple[int, str, str]:
    """Map an orchestrator failure detail to (status, error type, code).

    * KV page-pool exhaustion is CAPACITY (503 + Retry-After) — unless the
      engine's counts say the prompt cannot fit an EMPTY pool, which is
      client-caused (400, context_length_exceeded).
    * Rejected prompts (empty / out-of-vocab token ids) are 400.
    * Engine command timeouts are 504.
    * Anything else is an internal 500.
    """
    d = (detail or "").lower()
    if not d:
        return _ERR_INTERNAL
    if "pool exhausted" in d:
        need = _POOL_NEED_RE.search(d)
        total = _POOL_TOTAL_RE.search(d)
        if need and total and int(need.group(1)) > int(total.group(1)):
            return _ERR_TOO_LONG
        return _ERR_CAPACITY
    if "empty prompt" in d or "out of vocab" in d:
        return _ERR_BAD_PROMPT
    if "timeout waiting for" in d:
        return _ERR_TIMEOUT
    return _ERR_INTERNAL


def request_error_parts(
    detail: str,
) -> tuple[int, ErrorResponse, dict[str, str] | None]:
    """(status code, OpenAI error body, extra headers) for a failed
    request — shared by the JSON and SSE error paths."""
    status, type_, code = classify_request_error(detail)
    message = (detail or "").strip() or "internal engine error"
    body = ErrorResponse(error=ErrorDetail(
        message=message, type=type_, code=code))
    headers = {"Retry-After": "1"} if status == 503 else None
    return status, body, headers


def sse_error_event(detail: str) -> str:
    """SSE payload for a request that failed mid-stream.

    Shape follows vLLM's streamed errors (``data: {"error": {...}}``
    followed by ``data: [DONE]`` — ref/vllm/vllm/entrypoints/generate/
    base/serving.py::create_streaming_error_response): the response
    headers are long gone, so the error rides the event stream instead of
    a status code, and the object is byte-for-byte the non-streaming
    error body."""
    _, body, _ = request_error_parts(detail)
    return f"data: {json.dumps(body.model_dump())}\n\n"


# ---------------------------------------------------------------------------
# Request tracker: thread-safe bridge between async HTTP and sync orchestrator
# ---------------------------------------------------------------------------

@dataclass
class _PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    cancelled: bool = False
    logprob_results: list | None = None
    error: str = ""              # failure detail when finish_reason=="error"


class _RequestTracker:
    def __init__(self) -> None:
        self._pending: dict[int, _PendingRequest] = {}
        self._lock = threading.Lock()

    def register(self, request_id: int) -> _PendingRequest:
        pr = _PendingRequest()
        with self._lock:
            self._pending[request_id] = pr
        return pr

    def complete(
        self,
        request_id: int,
        token_ids: list[int],
        finish_reason: str,
        logprob_results: list | None = None,
        error: str = "",
    ) -> None:
        """on_complete sink.  ``error`` is the orchestrator's failure
        detail (TD-SERVE-ERROR-MASKING); it stays "" for the four-argument
        callers that predate it."""
        with self._lock:
            pr = self._pending.pop(request_id, None)
        if pr is not None:
            pr.token_ids = token_ids
            pr.finish_reason = finish_reason
            pr.logprob_results = logprob_results
            pr.error = error or ""
            pr.event.set()

    def cancel(self, request_id: int) -> None:
        with self._lock:
            pr = self._pending.pop(request_id, None)
        if pr is not None:
            pr.cancelled = True
            pr.event.set()


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------

class LayerStoRmServer:
    """OpenAI-compatible HTTP server backed by the successor Orchestrator (bridge core)."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        tokenizer: TokenizerWrapper,
        chat_template: ChatTemplateRenderer,
        metadata: EngineMetadata,
        *,
        model_name: str = "layerstorm",
        model_type: str = "",
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrent: int = 32,
        max_sequence_length: int = 32768,
        tool_call_parser: str = "",
        enable_auto_tool_choice: bool = False,
        reasoning_parser: str = "",
        reasoning_config: dict[str, Any] | None = None,
        tokenizer_mode: str = "",
        guided_manager: Any = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._tokenizer = tokenizer
        self._chat_template = chat_template
        self._metadata = metadata
        self._model_name = model_name
        self._host = host
        self._port = port
        self._max_concurrent = max_concurrent
        self._max_sequence_length = max_sequence_length

        self._next_request_id = itertools.count(1)
        self._tracker = _RequestTracker()
        self._tool_parser = get_tool_call_parser(model_type)

        # Named (vLLM-parity) parsers — fail fast on unknown names at boot.
        # When a named tool parser is configured it REPLACES the legacy
        # model_type parser and is gated by auto-tool-choice semantics; the
        # legacy always-parse behavior is kept only when no name is given.
        self._named_tool_parser_cls = (
            get_tool_parser(tool_call_parser) if tool_call_parser else None)
        if enable_auto_tool_choice and self._named_tool_parser_cls is None:
            raise ValueError(
                "enable_auto_tool_choice requires a tool_call_parser "
                "(--tool-call-parser / serving.tool_call_parser)")
        self._enable_auto_tool_choice = enable_auto_tool_choice
        # Serving tokenizer mode (vLLM TokenizerMode parity subset):
        # "deepseek_v4" switches the chat-template kwarg normalization +
        # thinking default to the DeepSeek-V4 rules (template defaults
        # thinking OFF; reasoning_effort none→off, max/xhigh→"max",
        # other→"high" — ref/vllm/vllm/tokenizers/deepseek_v4.py).
        # ""/"auto"/"hf" keep the legacy (GLM-shaped) behavior.
        self._tokenizer_mode = tokenizer_mode
        # --reasoning-config JSON passthrough (vLLM ReasoningConfig field
        # names): marker-string overrides for the named reasoning parser.
        rc = reasoning_config or {}
        self._reasoning_parser = (
            get_reasoning_parser(reasoning_parser)(
                metadata.think_start_token_id,
                metadata.think_end_token_id,
                reasoning_start_str=rc.get("reasoning_start_str"),
                reasoning_end_str=rc.get("reasoning_end_str"),
            ) if reasoning_parser else None)
        # Guided decoding (TD-SERVE-NAMED-TOOL-CHOICE): named tool_choice /
        # "required" compile the request's tool schemas into an xgrammar
        # grammar (glm47 wire format) — constrained requests carry a
        # per-request GuidedState into the orchestrator.  Built lazily on
        # first use; None when the tokenizer/vocab cannot support it
        # (those requests get a 400 with the reason).
        if guided_manager is not None:
            self._guided_manager = guided_manager
        else:
            hf = getattr(tokenizer, "hf_tokenizer", None)
            vocab = int(getattr(metadata, "vocab_size", 0) or 0)
            self._guided_manager = (
                GuidedDecodingManager(hf, vocab)
                if hf is not None and vocab > 0 else None)
        self._concurrency_lock = threading.Lock()
        self._active_requests = 0

        self._server_thread: threading.Thread | None = None
        self._uvicorn_server: uvicorn.Server | None = None

        self._app = self._create_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="LayerStoRm", version="0.1.0")

        app.add_api_route("/health", self._health, methods=["GET"])
        app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        app.add_api_route(
            "/v1/completions", self._handle_completions, methods=["POST"],
        )
        app.add_api_route(
            "/v1/chat/completions",
            self._handle_chat_completions,
            methods=["POST"],
        )
        return app

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def _health(self) -> Response:
        return Response(status_code=200)

    async def _list_models(self) -> ModelList:
        return ModelList(data=[ModelObject(id=self._model_name)])

    @staticmethod
    def _sampling_of(request) -> SamplingParams:
        """Map API sampling fields to SamplingParams.

        UNSPECIFIED temperature routes to the greedy champion
        (speculative) path — the pre-successor stack was always greedy,
        and OpenAI's implicit default of 1.0 must not silently move every
        request onto the slower sampled path.  An EXPLICIT temperature is
        honored: 0 = greedy, >0 = distribution-lossless sampled
        speculation (plain-arm fallback).  ``seed`` (OpenAI field) and
        ``top_k`` (extension) thread through when present — a fixed seed
        reproduces the token stream exactly (per-arm RNG streams; see
        SamplingParams)."""
        explicit = "temperature" in request.model_fields_set
        t = float(request.temperature) if explicit else 0.0
        kw = {}
        seed = getattr(request, "seed", None)
        if seed is not None:
            kw["seed"] = int(seed)
        top_k = getattr(request, "top_k", None)
        if top_k is not None:
            kw["top_k"] = int(top_k)
        return SamplingParams(temperature=max(0.0, t),
                              top_p=float(getattr(request, "top_p", 1.0)),
                              **kw)

    async def _handle_completions(
        self,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Response:
        acquired = False
        try:
            acquired = await self._try_acquire_semaphore()
            if not acquired:
                return self._error_response(
                    429, "Too many requests", code="rate_limit_exceeded",
                    headers={"Retry-After": "1"},
                )

            if isinstance(request.prompt, str):
                prompt_ids = self._tokenizer.encode(request.prompt)
            else:
                prompt_ids = list(request.prompt)

            if len(prompt_ids) > self._max_sequence_length:
                return self._error_response(
                    400,
                    f"Prompt length {len(prompt_ids)} exceeds "
                    f"max_sequence_length {self._max_sequence_length}",
                )

            request_id = next(self._next_request_id)
            max_tokens = request.max_tokens if request.max_tokens is not None else 0

            if request.stream:
                return await self._stream_completions(
                    request, raw_request, request_id, prompt_ids, max_tokens,
                )

            pending = self._tracker.register(request_id)
            inf_req = InferenceRequest(
                request_id=request_id,
                prompt_token_ids=prompt_ids,
                max_tokens=max_tokens,
                sampling=self._sampling_of(request),
                logprobs=request.logprobs,
                on_complete=self._tracker.complete,
            )
            self._orchestrator.submit_request(inf_req)

            result = await self._wait_for_result(
                request_id, pending, raw_request,
            )
            if result is None:
                return Response(status_code=499)

            token_ids, finish_reason = result
            if finish_reason == "error":
                return self._request_error_response(request_id, pending.error)
            token_ids = self._strip_eos(token_ids)
            text = self._tokenizer.decode(token_ids)

            if request.stop:
                text, was_stopped = _apply_stop_sequences(text, request.stop)
                if was_stopped:
                    finish_reason = "stop"

            lp = _build_completion_logprobs(
                pending.logprob_results, token_ids, self._tokenizer,
            ) if request.logprobs is not None and pending.logprob_results else None

            return JSONResponse(content=CompletionResponse(
                id=f"cmpl-{request_id}",
                created=int(time.time()),
                model=request.model,
                choices=[CompletionChoice(
                    text=text,
                    finish_reason=finish_reason,
                    logprobs=lp,
                )],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(token_ids),
                    total_tokens=len(prompt_ids) + len(token_ids),
                ),
            ).model_dump())

        finally:
            if acquired:
                self._release_semaphore()

    def _thinking_enabled(self, request: ChatCompletionRequest) -> bool:
        """Effective thinking switch for this request.

        Legacy modes: thinking defaults ON (GLM templates pre-seed
        ``<think>``), so only an explicit ``thinking: false`` disables the
        reasoning split.  deepseek_v4 mode: the V4 template defaults
        thinking OFF; it is enabled by ``thinking``/``enable_thinking``
        true and force-disabled by ``reasoning_effort: "none"``
        (ref/vllm/vllm/tokenizers/deepseek_v4.py normalization)."""
        if self._tokenizer_mode == "deepseek_v4":
            thinking = bool(request.thinking) or bool(
                getattr(request, "enable_thinking", False))
            if getattr(request, "reasoning_effort", None) == "none":
                return False
            return thinking
        return request.thinking is not False

    def _render_kwargs(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Chat-template kwargs per tokenizer mode.  deepseek_v4:
        normalized thinking (explicit, so template and parsers agree) +
        reasoning_effort mapping "max"/"xhigh"→"max", other→"high"
        ("none" disables thinking and is not forwarded).  Legacy: the
        historical passthrough (thinking only, None not injected)."""
        if self._tokenizer_mode == "deepseek_v4":
            kw: dict[str, Any] = {"thinking": self._thinking_enabled(request)}
            eff = getattr(request, "reasoning_effort", None)
            if isinstance(eff, str) and eff != "none":
                kw["reasoning_effort"] = ("max" if eff in ("max", "xhigh")
                                          else "high")
            return kw
        return {"thinking": request.thinking}

    @staticmethod
    def _chat_num_logprobs(request: ChatCompletionRequest) -> int | None:
        """OpenAI chat logprobs mapping: ``logprobs: true`` serves the
        chosen-token logprob per content token; ``top_logprobs`` (0-20,
        default 0) additionally sizes the per-token top list.  None =
        off (byte-identical decode, TD-ORCH-LOGPROBS)."""
        if not request.logprobs:
            return None
        return int(request.top_logprobs or 0)

    @staticmethod
    def _is_guided_choice(request: ChatCompletionRequest) -> bool:
        """True when tool_choice forces a call: a named function object or
        "required" — served via guided (grammar-constrained) decoding."""
        tc = request.tool_choice
        return isinstance(tc, dict) or tc == "required"

    def _tool_choice_error(
        self, request: ChatCompletionRequest,
    ) -> JSONResponse | None:
        """Semantic tool_choice validation (shape is checked by pydantic).

        Named tool_choice / "required" are served via GUIDED DECODING
        (TD-SERVE-NAMED-TOOL-CHOICE): validated here, compiled to a
        grammar in _build_guided.  "auto" with a named parser configured
        requires enable_auto_tool_choice (vLLM parity).
        """
        tc = request.tool_choice
        if tc is None or tc == "none":
            return None
        if isinstance(tc, dict) or tc == "required":
            if not request.tools:
                return self._error_response(
                    400,
                    f'tool_choice {tc if isinstance(tc, str) else "named"} '
                    "requires a non-empty tools list",
                    param="tool_choice")
            if isinstance(tc, dict):
                name = tc["function"]["name"]
                names = {
                    (t.get("function") or {}).get("name")
                    if isinstance(t.get("function"), dict) else t.get("name")
                    for t in request.tools if isinstance(t, dict)
                }
                if name not in names:
                    return self._error_response(
                        400,
                        f"tool_choice function '{name}' is not in tools",
                        param="tool_choice")
            if self._guided_manager is None:
                return self._error_response(
                    400,
                    "guided decoding unavailable (tokenizer/vocab not "
                    "wired) — named tool_choice / \"required\" need it",
                    param="tool_choice")
            return None
        # tc == "auto"
        if (request.tools and self._named_tool_parser_cls is not None
                and not self._enable_auto_tool_choice):
            return self._error_response(
                400,
                'tool_choice "auto" requires the server to be started '
                "with --enable-auto-tool-choice "
                "(serving.enable_auto_tool_choice)",
                param="tool_choice")
        return None

    def _build_guided(
        self, request: ChatCompletionRequest,
    ) -> tuple[Any, JSONResponse | None]:
        """Compile the guided-decoding grammar state for a forced
        tool_choice request; (None, None) for unconstrained requests,
        (None, 400) when grammar construction fails."""
        if not self._is_guided_choice(request):
            return None, None
        # reasoning=True ⇔ generation starts inside <think> (the template
        # pre-seeds "<think>"); non-thinking templates pre-seed a closed
        # think section so the grammar engages at position 0.
        reasoning = self._thinking_enabled(request)
        # Wire-format grammar follows the configured named tool parser
        # (glm47 → "glm_4_7", deepseek_v4 → "deepseek_v4"); the legacy
        # model_type route keeps the historical GLM wrapper.
        st_model = getattr(self._named_tool_parser_cls,
                           "structural_tag_model", "glm_4_7") \
            if self._named_tool_parser_cls is not None else "glm_4_7"
        try:
            grammar = self._guided_manager.build(
                request.tools, request.tool_choice, reasoning=reasoning,
                model=st_model)
            return grammar.new_state(), None
        except GuidedError as exc:
            return None, self._error_response(400, str(exc),
                                              param="tool_choice")

    def _should_parse_tools(self, request: ChatCompletionRequest) -> bool:
        """Auto-tool-choice gating.  Legacy behavior (no named parser
        configured): always parse with the model_type parser.  Named
        parser: run only when auto tool choice is enabled AND the request
        carries tools with tool_choice absent/"auto".  Guided (forced)
        requests ALWAYS parse — the output is grammar-forced into the tool
        wire format, so the parser must extract it regardless of the
        auto-tool-choice gate."""
        if request.tool_choice == "none":
            return False
        if self._is_guided_choice(request):
            return True
        if self._named_tool_parser_cls is None:
            return True
        if not self._enable_auto_tool_choice or not request.tools:
            return False
        return request.tool_choice in (None, "auto")

    def _split_reasoning(
        self, request: ChatCompletionRequest, token_ids: list[int],
    ) -> tuple[str | None, str]:
        """Split generated ids into (reasoning_content, content_text).

        Token-id boundary detection (metadata think ids) preferred; text
        markers only as fallback.  Thinking-disabled requests (per
        _thinking_enabled — mode-aware) skip the split: the template then
        pre-seeds a closed think section in the PROMPT and the output
        carries no markers.
        """
        if self._reasoning_parser is None or not self._thinking_enabled(
                request):
            return None, self._tokenizer.decode(token_ids)
        split = self._reasoning_parser.split_token_ids(token_ids)
        if split is not None:
            reasoning_ids, content_ids = split
            reasoning = self._tokenizer.decode(reasoning_ids) or None
            return reasoning, self._tokenizer.decode(content_ids)
        reasoning, content = self._reasoning_parser.extract_reasoning(
            self._tokenizer.decode(token_ids))
        return reasoning, content or ""

    async def _handle_chat_completions(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Response:
        acquired = False
        try:
            acquired = await self._try_acquire_semaphore()
            if not acquired:
                return self._error_response(
                    429, "Too many requests", code="rate_limit_exceeded",
                    headers={"Retry-After": "1"},
                )

            err = self._tool_choice_error(request)
            if err is not None:
                return err
            guided_state, gerr = self._build_guided(request)
            if gerr is not None:
                return gerr

            prompt_text = self._chat_template.render(
                request.messages,
                tools=request.tools,
                **self._render_kwargs(request),
            )
            prompt_ids = self._tokenizer.encode(prompt_text)

            if len(prompt_ids) > self._max_sequence_length:
                return self._error_response(
                    400,
                    f"Prompt length {len(prompt_ids)} exceeds "
                    f"max_sequence_length {self._max_sequence_length}",
                )

            request_id = next(self._next_request_id)
            max_tokens = request.max_tokens if request.max_tokens is not None else 0

            if request.stream:
                return await self._stream_chat_completions(
                    request, raw_request, request_id, prompt_ids, max_tokens,
                    guided_state,
                )

            num_lp = self._chat_num_logprobs(request)
            pending = self._tracker.register(request_id)
            inf_req = InferenceRequest(
                request_id=request_id,
                prompt_token_ids=prompt_ids,
                max_tokens=max_tokens,
                sampling=self._sampling_of(request),
                logprobs=num_lp,
                on_complete=self._tracker.complete,
                guided=guided_state,
                force_plain=bool(getattr(request, "ls_force_plain", False)),
            )
            self._orchestrator.submit_request(inf_req)

            result = await self._wait_for_result(
                request_id, pending, raw_request,
            )
            if result is None:
                return Response(status_code=499)

            token_ids, finish_reason = result
            if finish_reason == "error":
                return self._request_error_response(request_id, pending.error)
            token_ids = self._strip_eos(token_ids)
            reasoning_content, text = self._split_reasoning(
                request, token_ids)

            if request.stop:
                text, was_stopped = _apply_stop_sequences(text, request.stop)
                if was_stopped:
                    finish_reason = "stop"

            tool_calls_resp: list[ToolCallResponse] | None = None
            content: str | None = text
            if self._should_parse_tools(request):
                if self._named_tool_parser_cls is not None:
                    parsed = self._named_tool_parser_cls(
                        tools=request.tools).extract_tool_calls(text)
                else:
                    parsed = self._tool_parser.parse(text)
                if parsed.tool_calls:
                    finish_reason = "tool_calls"
                    content = parsed.content
                    tool_calls_resp = [
                        ToolCallResponse(
                            id=tc.id,
                            function=ToolCallFunctionResponse(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                        for tc in parsed.tool_calls
                    ]
            if not content and reasoning_content:
                content = None

            chat_lp = _build_chat_logprobs(
                pending.logprob_results, token_ids, self._tokenizer,
            ) if num_lp is not None and pending.logprob_results else None

            return JSONResponse(content=ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(
                        content=content,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls_resp,
                    ),
                    finish_reason=finish_reason,
                    logprobs=chat_lp,
                )],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(token_ids),
                    total_tokens=len(prompt_ids) + len(token_ids),
                ),
            ).model_dump())

        finally:
            if acquired:
                self._release_semaphore()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _stream_completions(
        self,
        request: CompletionRequest,
        raw_request: Request,
        request_id: int,
        prompt_ids: list[int],
        max_tokens: int,
    ) -> StreamingResponse:
        from server.streaming import TokenQueue, stream_completion_response

        tq = TokenQueue()
        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            max_tokens=max_tokens,
            sampling=self._sampling_of(request),
            logprobs=request.logprobs,
            on_token=tq.push,
            on_complete=tq.mark_done,
        )
        self._orchestrator.submit_request(inf_req)

        def _cancel(rid: int) -> None:
            self._orchestrator.cancel_request(rid)

        gen = stream_completion_response(
            request_id=request_id,
            response_id=f"cmpl-{request_id}",
            model=request.model,
            token_queue=tq,
            tokenizer=self._tokenizer,
            eos_token_ids=self._metadata.eos_token_ids,
            raw_request=raw_request,
            cancel_fn=_cancel,
            stop=request.stop,
            prompt_tokens=len(prompt_ids),
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    async def _stream_chat_completions(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        request_id: int,
        prompt_ids: list[int],
        max_tokens: int,
        guided_state: Any = None,
    ) -> StreamingResponse:
        from server.streaming import TokenQueue, stream_chat_completion_response

        num_lp = self._chat_num_logprobs(request)
        tq = TokenQueue()
        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            max_tokens=max_tokens,
            sampling=self._sampling_of(request),
            logprobs=num_lp,
            on_token=tq.push,
            on_complete=tq.mark_done,
            guided=guided_state,
            force_plain=bool(getattr(request, "ls_force_plain", False)),
        )
        self._orchestrator.submit_request(inf_req)

        def _cancel(rid: int) -> None:
            self._orchestrator.cancel_request(rid)

        # Parser wiring: a configured named tool parser replaces the legacy
        # whole-text final-chunk parse with true incremental deltas (gated
        # by auto tool choice); the legacy model_type parser is kept only
        # when no name is configured.
        run_tools = self._should_parse_tools(request)
        if self._named_tool_parser_cls is not None:
            legacy_tool_parser = None
            tool_stream = (self._named_tool_parser_cls(
                tools=request.tools).stream() if run_tools else None)
        else:
            legacy_tool_parser = self._tool_parser if run_tools else None
            tool_stream = None
        reasoning_stream = (
            self._reasoning_parser.stream(
                thinking=self._thinking_enabled(request))
            if self._reasoning_parser is not None else None)

        gen = stream_chat_completion_response(
            request_id=request_id,
            response_id=f"chatcmpl-{request_id}",
            model=request.model,
            token_queue=tq,
            tokenizer=self._tokenizer,
            eos_token_ids=self._metadata.eos_token_ids,
            tool_parser=legacy_tool_parser,
            raw_request=raw_request,
            cancel_fn=_cancel,
            stop=request.stop,
            prompt_tokens=len(prompt_ids),
            reasoning_stream=reasoning_stream,
            tool_stream=tool_stream,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    async def _try_acquire_semaphore(self) -> bool:
        with self._concurrency_lock:
            if self._active_requests >= self._max_concurrent:
                return False
            self._active_requests += 1
            return True

    def _release_semaphore(self) -> None:
        with self._concurrency_lock:
            self._active_requests -= 1

    async def _wait_for_result(
        self,
        request_id: int,
        pending: _PendingRequest,
        raw_request: Request,
    ) -> tuple[list[int], str] | None:
        loop = asyncio.get_event_loop()
        while True:
            done = await loop.run_in_executor(
                None, pending.event.wait, 0.1,
            )
            if done:
                if pending.cancelled:
                    return None
                return (pending.token_ids, pending.finish_reason)
            if await raw_request.is_disconnected():
                self._orchestrator.cancel_request(request_id)
                self._tracker.cancel(request_id)
                return None

    def _strip_eos(self, token_ids: list[int]) -> list[int]:
        eos = self._metadata.eos_token_ids
        if not eos or not token_ids:
            return token_ids
        while token_ids and token_ids[-1] in eos:
            token_ids = token_ids[:-1]
        return token_ids

    @staticmethod
    def _request_error_response(
        request_id: int, detail: str,
    ) -> JSONResponse:
        """OpenAI-shaped error for a request the orchestrator FAILED
        (finish_reason "error").  Never a 200 with an empty completion —
        the client must be able to tell "the model emitted nothing" from
        "the request died" (TD-SERVE-ERROR-MASKING / INV-SERVE-ERROR)."""
        status, body, headers = request_error_parts(detail)
        log.error("request %d failed (%d): %s", request_id, status,
                  body.error.message)
        return JSONResponse(status_code=status, content=body.model_dump(),
                            headers=headers)

    @staticmethod
    def _error_response(
        status_code: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        body = ErrorResponse(error=ErrorDetail(
            message=message, type=type_, code=code, param=param,
        ))
        return JSONResponse(
            status_code=status_code,
            content=body.model_dump(),
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._server_thread = threading.Thread(
            target=self._run_uvicorn, daemon=True,
        )
        self._server_thread.start()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until uvicorn accepts connections (or the thread died).

        The serve entrypoint (#71) calls this after ``run()`` so a bind
        failure (port in use, bad address) surfaces at boot instead of the
        orchestrator loop spinning behind a dead server.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            srv = self._uvicorn_server
            if srv is not None and srv.started:
                return True
            if (self._server_thread is not None
                    and not self._server_thread.is_alive()):
                return False  # uvicorn thread exited before starting
            time.sleep(0.02)
        return False

    @property
    def bound_port(self) -> int | None:
        """Actual bound port (resolves port=0 ephemeral binds); None if
        the server is not (yet) listening."""
        srv = self._uvicorn_server
        if srv is None or not srv.started:
            return None
        for s in getattr(srv, "servers", []):
            for sock in getattr(s, "sockets", []):
                try:
                    return int(sock.getsockname()[1])
                except (OSError, IndexError, TypeError):
                    continue
        return None

    def _run_uvicorn(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()

    def shutdown(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _apply_stop_sequences(
    text: str, stop: str | list[str],
) -> tuple[str, bool]:
    if isinstance(stop, str):
        stop = [stop]
    earliest = len(text)
    for seq in stop:
        idx = text.find(seq)
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest < len(text):
        return text[:earliest], True
    return text, False


def _build_completion_logprobs(
    step_results: list,
    token_ids: list[int],
    tokenizer: Any,
) -> CompletionLogprobs | None:
    """OpenAI completions logprobs block from per-token StepLogprobs.

    Iterates ``token_ids`` (the SERVED tokens — EOS-stripped), so a
    step_results list longer than the response (trailing EOS entries)
    stays aligned."""
    if not step_results:
        return None
    text_offset: list[int] = []
    token_logprobs: list[float | None] = []
    tokens: list[str] = []
    top_logprobs: list[dict[str, float] | None] = []
    offset = 0
    for i, tid in enumerate(token_ids):
        step = step_results[i] if i < len(step_results) else None
        tok_str = tokenizer.decode([tid])
        tokens.append(tok_str)
        text_offset.append(offset)
        offset += len(tok_str)
        if step is None:
            token_logprobs.append(None)
            top_logprobs.append(None)
        else:
            token_logprobs.append(step.token.logprob)
            top_dict: dict[str, float] = {}
            for tlp in step.top_logprobs:
                top_dict[tokenizer.decode([tlp.token_id])] = tlp.logprob
            top_logprobs.append(top_dict if top_dict else None)
    return CompletionLogprobs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs,
    )


def _build_chat_logprobs(
    step_results: list,
    token_ids: list[int],
    tokenizer: Any,
) -> ChatCompletionLogprobs | None:
    """OpenAI chat logprobs.content block; aligned over ``token_ids``
    (the served, EOS-stripped tokens) like _build_completion_logprobs."""
    if not step_results:
        return None
    content: list[ChatCompletionTokenLogprobContent] = []
    for i, tid in enumerate(token_ids):
        step = step_results[i] if i < len(step_results) else None
        tok_str = tokenizer.decode([tid])
        tok_bytes = list(tok_str.encode("utf-8"))
        if step is None:
            content.append(ChatCompletionTokenLogprobContent(
                token=tok_str,
                logprob=-9999.0,
                bytes=tok_bytes,
                top_logprobs=[],
            ))
        else:
            tops = [
                ChatCompletionTokenLogprob(
                    token=tokenizer.decode([tlp.token_id]),
                    logprob=tlp.logprob,
                    bytes=list(tokenizer.decode([tlp.token_id]).encode("utf-8")),
                )
                for tlp in step.top_logprobs
            ]
            content.append(ChatCompletionTokenLogprobContent(
                token=tok_str,
                logprob=step.token.logprob,
                bytes=tok_bytes,
                top_logprobs=tops,
            ))
    return ChatCompletionLogprobs(content=content)
