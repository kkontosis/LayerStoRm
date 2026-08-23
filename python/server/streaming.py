"""SSE streaming response generators for token-by-token output.

Produces Server-Sent Events in OpenAI-compatible format::

    data: {"id":"cmpl-1","object":"text_completion","choices":[{"text":"Hello","index":0}]}\n\n
    data: {"id":"cmpl-1","object":"text_completion","choices":[{"text":" world","index":0,"finish_reason":"stop"}],"usage":{...}}\n\n
    data: [DONE]\n\n
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from pydantic import BaseModel

from server.http_server import (
    ChatCompletionLogprobs,
    CompletionLogprobs,
    UsageInfo,
    _apply_stop_sequences,
    _build_chat_logprobs,
    _build_completion_logprobs,
    sse_error_event,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming response models
# ---------------------------------------------------------------------------

class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None
    logprobs: CompletionLogprobs | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]
    usage: UsageInfo | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageInfo | None = None


# ---------------------------------------------------------------------------
# Token queue used by streaming requests
# ---------------------------------------------------------------------------

class TokenQueue:
    """Thread-safe token queue for streaming.

    The orchestrator's ``on_token`` callback pushes token IDs from the
    orchestrator thread.  The SSE async generator drains them from the
    HTTP server's event loop via ``drain()``.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[int, Any]] = queue.Queue()
        self._done = False
        self._finish_reason = "stop"
        self._error = ""

    def push(self, request_id: int, token_id: int, step_logprobs: Any = None) -> None:
        self._queue.put((token_id, step_logprobs))

    def mark_done(
        self, request_id: int, token_ids: list[int], finish_reason: str,
        logprob_results: Any = None, error: str = "",
    ) -> None:
        """on_complete sink.  ``error`` carries the orchestrator's failure
        detail when ``finish_reason`` is "error" (TD-SERVE-ERROR-MASKING);
        four-argument callers keep working and leave it ""."""
        # Publish the outcome BEFORE the done flag: the SSE generator polls
        # ``done`` and then reads finish_reason/error, so the reverse order
        # could show it a completed stream with a stale reason.
        self._finish_reason = finish_reason
        self._error = error or ""
        self._done = True

    def drain(self) -> list[tuple[int, Any]]:
        items: list[tuple[int, Any]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    @property
    def done(self) -> bool:
        return self._done

    @property
    def finish_reason(self) -> str:
        return self._finish_reason

    @property
    def failed(self) -> bool:
        """True once the orchestrator reported this request as FAILED."""
        return self._finish_reason == "error"

    @property
    def error(self) -> str:
        return self._error


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_line(data: str) -> str:
    return f"data: {data}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _completion_lp_fragment(
    pairs: list[tuple[int, Any]], tokenizer: Any, base_offset: int,
) -> tuple[CompletionLogprobs | None, int]:
    """Per-chunk logprobs block for one drained batch of (token_id,
    StepLogprobs|None) pairs.  All-None pairs (logprobs-off requests)
    return (None, base_offset) — the chunk serializes byte-identically to
    today.  ``base_offset`` keeps text_offset absolute across chunks."""
    if not any(lp is not None for _, lp in pairs):
        return None, base_offset
    frag = _build_completion_logprobs(
        [lp for _, lp in pairs], [t for t, _ in pairs], tokenizer)
    frag.text_offset = [o + base_offset for o in frag.text_offset]
    return frag, base_offset + sum(len(t) for t in frag.tokens)


def _chat_lp_fragment(
    pairs: list[tuple[int, Any]], tokenizer: Any,
) -> ChatCompletionLogprobs | None:
    """Per-chunk chat logprobs.content block; None for all-None pairs
    (logprobs-off requests keep today's byte-identical chunks)."""
    if not any(lp is not None for _, lp in pairs):
        return None
    return _build_chat_logprobs(
        [lp for _, lp in pairs], [t for t, _ in pairs], tokenizer)


# ---------------------------------------------------------------------------
# Completion stream generator
# ---------------------------------------------------------------------------

async def stream_completion_response(
    *,
    request_id: int,
    response_id: str,
    model: str,
    token_queue: TokenQueue,
    tokenizer: Any,
    eos_token_ids: tuple[int, ...],
    raw_request: Request,
    cancel_fn: Any,
    stop: str | list[str] | None = None,
    prompt_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE strings for ``/v1/completions``."""
    created = int(time.time())
    loop = asyncio.get_event_loop()
    all_tokens: list[int] = []
    prev_text = ""
    stopped = False
    lp_offset = 0                    # absolute text_offset across chunks

    while True:
        new_items = await loop.run_in_executor(None, token_queue.drain)

        if new_items:
            pairs: list[tuple[int, Any]] = []
            for tok, lp in new_items:
                if tok in eos_token_ids:
                    continue
                all_tokens.append(tok)
                pairs.append((tok, lp))

            full_text = tokenizer.decode(all_tokens)
            delta_text = full_text[len(prev_text):]

            if stop and delta_text:
                truncated, was_stopped = _apply_stop_sequences(
                    full_text, stop,
                )
                if was_stopped:
                    delta_text = truncated[len(prev_text):]
                    prev_text = truncated
                    stopped = True
                else:
                    prev_text = full_text
            else:
                prev_text = full_text

            lp_frag, lp_offset = _completion_lp_fragment(
                pairs, tokenizer, lp_offset)
            if delta_text or lp_frag is not None:
                chunk = CompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[CompletionStreamChoice(text=delta_text,
                                                    logprobs=lp_frag)],
                )
                yield _sse_line(chunk.model_dump_json(exclude_none=True))

            if stopped:
                final = CompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[CompletionStreamChoice(
                        text="", finish_reason="stop",
                    )],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(all_tokens),
                        total_tokens=prompt_tokens + len(all_tokens),
                    ),
                )
                yield _sse_line(final.model_dump_json(exclude_none=True))
                yield _sse_done()
                return

        if token_queue.done:
            if token_queue.failed:
                # TD-SERVE-ERROR-MASKING: a failed request terminates the
                # stream with the OpenAI error object — never with a
                # fabricated finish_reason chunk.  Pending deltas are
                # dropped: the response is void, not truncated.
                log.error("stream request %d failed: %s", request_id,
                          token_queue.error)
                yield sse_error_event(token_queue.error)
                yield _sse_done()
                return
            remaining = await loop.run_in_executor(None, token_queue.drain)
            pairs = []
            for tok, lp in remaining:
                if tok not in eos_token_ids:
                    all_tokens.append(tok)
                    pairs.append((tok, lp))
            if remaining:
                full_text = tokenizer.decode(all_tokens)
                delta_text = full_text[len(prev_text):]
                if stop and delta_text:
                    truncated, was_stopped = _apply_stop_sequences(
                        full_text, stop,
                    )
                    if was_stopped:
                        delta_text = truncated[len(prev_text):]
                lp_frag, lp_offset = _completion_lp_fragment(
                    pairs, tokenizer, lp_offset)
                if delta_text or lp_frag is not None:
                    chunk = CompletionStreamResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        choices=[CompletionStreamChoice(text=delta_text,
                                                        logprobs=lp_frag)],
                    )
                    yield _sse_line(chunk.model_dump_json(exclude_none=True))

            final = CompletionStreamResponse(
                id=response_id,
                created=created,
                model=model,
                choices=[CompletionStreamChoice(
                    text="",
                    finish_reason=token_queue.finish_reason,
                )],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(all_tokens),
                    total_tokens=prompt_tokens + len(all_tokens),
                ),
            )
            yield _sse_line(final.model_dump_json(exclude_none=True))
            yield _sse_done()
            return

        if await raw_request.is_disconnected():
            cancel_fn(request_id)
            return

        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Chat completion stream generator
# ---------------------------------------------------------------------------

async def stream_chat_completion_response(
    *,
    request_id: int,
    response_id: str,
    model: str,
    token_queue: TokenQueue,
    tokenizer: Any,
    eos_token_ids: tuple[int, ...],
    tool_parser: Any,
    raw_request: Request,
    cancel_fn: Any,
    stop: str | list[str] | None = None,
    prompt_tokens: int = 0,
    reasoning_stream: Any = None,
    tool_stream: Any = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE strings for ``/v1/chat/completions``.

    ``reasoning_stream`` (server.reasoning_parsers.ReasoningStream) routes
    decoded deltas into ``delta.reasoning_content`` vs ``delta.content``
    using token-id think boundaries.  ``tool_stream``
    (server.tool_parsers.Glm47ToolStream) incrementally extracts OpenAI
    tool-call deltas from the content channel.  ``tool_parser`` is the
    LEGACY whole-text parser applied to the final chunk when no named
    streaming parser is configured (pass None to disable).
    """
    created = int(time.time())
    loop = asyncio.get_event_loop()
    all_tokens: list[int] = []
    prev_text = ""
    first_chunk = True
    stopped = False

    def _route(delta_text: str, delta_ids: list[int]) -> list[DeltaMessage]:
        """Split a raw decoded delta into reasoning / content / tool-call
        delta messages (empty pieces are dropped)."""
        msgs: list[DeltaMessage] = []
        if reasoning_stream is not None:
            r_delta, c_delta = reasoning_stream.process(delta_text, delta_ids)
        else:
            r_delta, c_delta = "", delta_text
        if r_delta:
            msgs.append(DeltaMessage(reasoning_content=r_delta))
        if c_delta:
            if tool_stream is not None:
                content, tc_deltas = tool_stream.process(c_delta)
                if content:
                    msgs.append(DeltaMessage(content=content))
                if tc_deltas:
                    msgs.append(DeltaMessage(tool_calls=tc_deltas))
            else:
                msgs.append(DeltaMessage(content=c_delta))
        return msgs

    def _flush() -> list[DeltaMessage]:
        """End-of-stream flush of the incremental parser states."""
        msgs: list[DeltaMessage] = []
        pending_content = ""
        if reasoning_stream is not None:
            r_tail, c_tail = reasoning_stream.finish()
            if r_tail:
                msgs.append(DeltaMessage(reasoning_content=r_tail))
            pending_content = c_tail
        if tool_stream is not None:
            content, tc_deltas = tool_stream.process(pending_content) \
                if pending_content else ("", [])
            f_content, f_deltas = tool_stream.finish()
            content += f_content
            tc_deltas = tc_deltas + f_deltas
            if content:
                msgs.append(DeltaMessage(content=content))
            if tc_deltas:
                msgs.append(DeltaMessage(tool_calls=tc_deltas))
        elif pending_content:
            msgs.append(DeltaMessage(content=pending_content))
        return msgs

    def _chunk(delta: DeltaMessage,
               finish_reason: str | None = None,
               usage: UsageInfo | None = None,
               logprobs: ChatCompletionLogprobs | None = None) -> str:
        resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[ChatCompletionStreamChoice(
                delta=delta, finish_reason=finish_reason,
                logprobs=logprobs,
            )],
            usage=usage,
        )
        return _sse_line(resp.model_dump_json(exclude_none=True))

    def _yield_routed(delta_text: str, delta_ids: list[int],
                      batch_lp: ChatCompletionLogprobs | None) -> list[str]:
        """Route one drained batch into delta chunks; the batch's
        logprobs block rides the FIRST chunk (a bare chunk is emitted
        when routing produced no text — e.g. an empty decode delta)."""
        msgs = _route(delta_text, delta_ids)
        if not msgs and batch_lp is not None:
            msgs = [DeltaMessage()]
        return [_chunk(msg, logprobs=batch_lp if k == 0 else None)
                for k, msg in enumerate(msgs)]

    while True:
        new_items = await loop.run_in_executor(None, token_queue.drain)

        if new_items:
            delta_ids: list[int] = []
            lp_pairs: list[tuple[int, Any]] = []
            for tok, lp in new_items:
                if tok in eos_token_ids:
                    continue
                all_tokens.append(tok)
                delta_ids.append(tok)
                lp_pairs.append((tok, lp))

            full_text = tokenizer.decode(all_tokens)
            delta_text = full_text[len(prev_text):]

            if stop and delta_text:
                truncated, was_stopped = _apply_stop_sequences(
                    full_text, stop,
                )
                if was_stopped:
                    delta_text = truncated[len(prev_text):]
                    prev_text = truncated
                    stopped = True
                else:
                    prev_text = full_text
            else:
                prev_text = full_text

            if first_chunk:
                yield _chunk(DeltaMessage(role="assistant", content=""))
                first_chunk = False

            for line in _yield_routed(delta_text, delta_ids,
                                      _chat_lp_fragment(lp_pairs,
                                                        tokenizer)):
                yield line

            if stopped:
                for msg in _flush():
                    yield _chunk(msg)
                yield _chunk(
                    DeltaMessage(),
                    finish_reason="stop",
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(all_tokens),
                        total_tokens=prompt_tokens + len(all_tokens),
                    ),
                )
                yield _sse_done()
                return

        if token_queue.done:
            if token_queue.failed:
                # TD-SERVE-ERROR-MASKING: terminate with the error object
                # (no role/flush/finish chunks — the parser state may be
                # mid-tool-call and the response is void, not truncated).
                log.error("stream request %d failed: %s", request_id,
                          token_queue.error)
                yield sse_error_event(token_queue.error)
                yield _sse_done()
                return
            remaining = await loop.run_in_executor(None, token_queue.drain)
            delta_ids = []
            lp_pairs = []
            for tok, lp in remaining:
                if tok not in eos_token_ids:
                    all_tokens.append(tok)
                    delta_ids.append(tok)
                    lp_pairs.append((tok, lp))

            full_text = tokenizer.decode(all_tokens)
            delta_text = full_text[len(prev_text):]

            if stop and delta_text:
                truncated, was_stopped = _apply_stop_sequences(
                    full_text, stop,
                )
                if was_stopped:
                    delta_text = truncated[len(prev_text):]

            if first_chunk:
                yield _chunk(DeltaMessage(role="assistant", content=""))
                first_chunk = False

            for line in _yield_routed(delta_text, delta_ids,
                                      _chat_lp_fragment(lp_pairs,
                                                        tokenizer)):
                yield line
            for msg in _flush():
                yield _chunk(msg)

            finish_reason = token_queue.finish_reason
            tool_calls_list: list[dict[str, Any]] | None = None
            if tool_stream is not None:
                if tool_stream.tools_called:
                    finish_reason = "tool_calls"
            elif tool_parser is not None:
                # Legacy path: whole-text parse, calls in the final chunk.
                parsed = tool_parser.parse(full_text)
                if parsed.tool_calls:
                    finish_reason = "tool_calls"
                    tool_calls_list = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in parsed.tool_calls
                    ]

            yield _chunk(
                DeltaMessage(tool_calls=tool_calls_list),
                finish_reason=finish_reason,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(all_tokens),
                    total_tokens=prompt_tokens + len(all_tokens),
                ),
            )
            yield _sse_done()
            return

        if await raw_request.is_disconnected():
            cancel_fn(request_id)
            return

        await asyncio.sleep(0.01)
