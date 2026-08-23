"""Named tool-call parsers — OpenAI ``tool_calls`` extraction, both modes.

Portions ported (lean) from vLLM (ref/vllm; Apache-2.0, Copyright
contributors to the vLLM project — see THIRD_PARTY_NOTICES.md).

Registry pattern mirroring vLLM's ToolParserManager, kept lean: parsers are
selected by CLI/config name (``serving.tool_call_parser`` /
``--tool-call-parser``) and instantiated PER REQUEST with the request's
``tools`` so argument values can be coerced against the declared JSON
schemas.

``glm47`` parses the GLM-4.x/5.x XML-ish wire format (emitted verbatim by
test-data/GLM-5.2/chat_template.jinja line 35):

    <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>

``deepseek_v4`` parses the DeepSeek-V4 DSML wire format (see its section
below; ported from ref/vllm/vllm/parser/deepseek_v4.py).

Ported (lean) from ref/vllm:
  - ref/vllm/vllm/parser/glm47_moe.py            (format, arg regexes,
    name strip, whitespace-preserving arg values)
  - ref/vllm/vllm/tool_parsers/utils.py           (extract_types_from_schema
    + coerce_to_schema_type: schema-driven typing; no schema → values stay
    strings)
  - ref/vllm/vllm/parser/engine/parser_engine.py  (streaming policy: only
    schema-string-only fields stream inside the value; other types are
    buffered to value close because their serialized form can change)

Streaming contract (OpenAI SSE): per tool call index, the first delta
carries ``{index, id, type, function:{name, arguments:""}}``; subsequent
deltas carry ``{index, function:{arguments: fragment}}`` and the
concatenated fragments always form the exact JSON object the non-streaming
parser would return.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from tokenizer.tool_call_parser import (
    ParsedToolCalls,
    ToolCall,
    ToolCallFunction,
)

__all__ = [
    "DeepSeekV4ToolParser",
    "DeepSeekV4ToolStream",
    "Glm47ToolParser",
    "Glm47ToolStream",
    "get_tool_parser",
    "tool_parser_names",
]

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_ARG_KEY_START = "<arg_key>"
_ARG_KEY_END = "</arg_key>"
_ARG_VALUE_START = "<arg_value>"
_ARG_VALUE_END = "</arg_value>"

_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)
# From ref/vllm/vllm/parser/glm47_moe.py (_ARG_RE): \s* between tags,
# value whitespace preserved.
_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Schema-driven value typing
# (port of ref/vllm/vllm/tool_parsers/utils.py — trimmed to what GLM needs)
# ---------------------------------------------------------------------------

_TYPE_ALIASES: dict[str, str] = {
    "str": "string", "text": "string", "enum": "string",
    "int": "integer", "int32": "integer", "int64": "integer",
    "long": "integer", "short": "integer", "unsigned": "integer",
    "float": "number", "float32": "number", "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array", "list": "array", "sequence": "array",
}


def _extract_types(schema: Any) -> list[str]:
    """All possible JSON Schema types for a property; ``["string"]`` when
    unknown (port of extract_types_from_schema)."""
    if not isinstance(schema, dict):
        return ["string"]
    types: set[str] = set()
    tv = schema.get("type")
    if isinstance(tv, str):
        types.add(tv)
    elif isinstance(tv, list):
        types.update(t for t in tv if isinstance(t, str))
    enum = schema.get("enum")
    if isinstance(enum, list):
        for v in enum:
            if v is None:
                types.add("null")
            elif isinstance(v, bool):
                types.add("boolean")
            elif isinstance(v, int):
                types.add("integer")
            elif isinstance(v, float):
                types.add("number")
            elif isinstance(v, str):
                types.add("string")
            elif isinstance(v, list):
                types.add("array")
            elif isinstance(v, dict):
                types.add("object")
    for choice in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(choice), list):
            for sub in schema[choice]:
                types.update(_extract_types(sub))
    return list(types) if types else ["string"]


def _json_finite(value: Any) -> bool:
    """True when json.dumps(value) yields valid JSON (no inf/nan)."""
    try:
        json.dumps(value, allow_nan=False)
        return True
    except ValueError:
        return False


def _coerce(value: str, schema_types: list[str]) -> Any:
    """Best-effort coercion of a raw string to a schema type
    (port of coerce_to_schema_type, same priority order)."""
    normalized = {_TYPE_ALIASES.get(t.strip().lower(), t.strip().lower())
                  for t in schema_types}
    for candidate in ("null", "integer", "number", "boolean",
                      "object", "array", "string"):
        if candidate not in normalized:
            continue
        if candidate == "null":
            if value.lower() == "null":
                return None
        elif candidate == "string":
            return value
        elif candidate == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        elif candidate == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            return int(val) if val == int(val) else val
        elif candidate == "boolean":
            low = value.lower().strip()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
        else:  # object / array
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _json_finite(parsed):
                return parsed
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    return parsed if _json_finite(parsed) else value


def _tool_properties(
    tools: list[dict[str, Any]] | None, func_name: str,
) -> dict[str, Any] | None:
    """Find the parameters.properties schema dict for ``func_name``.
    Accepts OpenAI-shaped tools ({"type":"function","function":{...}}) and
    bare function dicts."""
    if not tools:
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) \
            else tool
        if fn.get("name") != func_name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                return props
        return {}
    return None


def _json_escape(chunk: str) -> str:
    """JSON string escaping of a fragment (context-free per character, so
    safe to apply chunk-by-chunk while streaming a string value)."""
    return json.dumps(chunk, ensure_ascii=False)[1:-1]


def _partial_tag_len(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of
    ``tag`` — text that must be withheld until the next delta decides."""
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if tag.startswith(text[-k:]):
            return k
    return 0


# ---------------------------------------------------------------------------
# glm47 — non-streaming
# ---------------------------------------------------------------------------

class Glm47ToolParser:
    """GLM-4.7 wire-format tool-call parser (per-request instance)."""

    name = "glm47"
    # xgrammar builtin structural-tag model used by guided decoding
    # (named tool_choice / "required") when this parser is configured.
    structural_tag_model = "glm_4_7"

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self._tools = tools

    def _coerce_arg(self, func_name: str, key: str, raw: str) -> Any:
        props = _tool_properties(self._tools, func_name)
        if props is None:
            # No schema for this function: values stay strings
            # (ref/vllm parser_engine._fix_arg_types without-schema rule).
            return raw
        return _coerce(raw, _extract_types(props.get(key)))

    def extract_tool_calls(self, text: str) -> ParsedToolCalls:
        matches = list(_CALL_RE.finditer(text))
        if not matches:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        calls: list[ToolCall] = []
        for i, m in enumerate(matches):
            body = m.group("body")
            key_idx = body.find(_ARG_KEY_START)
            if key_idx >= 0:
                name = body[:key_idx].strip()
                args_body = body[key_idx:]
            else:
                name = body.strip()
                args_body = ""
            params: dict[str, Any] = {}
            for am in _ARG_RE.finditer(args_body):
                key = am.group("key").strip()
                params[key] = self._coerce_arg(name, key, am.group("value"))
            calls.append(ToolCall(
                id=f"call_{i}",
                function=ToolCallFunction(
                    name=name,
                    arguments=json.dumps(params, ensure_ascii=False),
                ),
            ))
        content = text[: matches[0].start()].strip()
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content or None,
        )

    def stream(self) -> "Glm47ToolStream":
        return Glm47ToolStream(self)


# ---------------------------------------------------------------------------
# glm47 — streaming
# ---------------------------------------------------------------------------

class Glm47ToolStream:
    """Incremental GLM-4.7 tool-call extractor.

    Feed decoded text deltas through ``process``; returns plain-content
    deltas plus OpenAI tool-call delta dicts.  ``finish()`` flushes withheld
    text and closes any open arguments JSON so the concatenated fragments
    are always valid.
    """

    def __init__(self, parser: Glm47ToolParser) -> None:
        self._parser = parser
        self._buf = ""
        self._state = "content"
        self._index = -1
        self._name = ""
        self._cur_key = ""
        self._cur_types: list[str] = []
        self._val_streaming = False
        self._args_open = False       # '{' fragment already emitted
        self._called = False

    # ── public API ──────────────────────────────────────────────────────

    @property
    def tools_called(self) -> bool:
        return self._called

    def process(self, delta_text: str) -> tuple[str, list[dict[str, Any]]]:
        """Returns ``(content_delta, tool_call_deltas)``."""
        self._buf += delta_text
        content: list[str] = []
        deltas: list[dict[str, Any]] = []
        while self._step(content, deltas):
            pass
        return "".join(content), deltas

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        """End-of-stream flush: withheld content is emitted; an open call's
        arguments JSON is closed so concatenation stays valid.  A call cut
        before its name completed never materialized and is dropped."""
        content: list[str] = []
        deltas: list[dict[str, Any]] = []
        state, buf = self._state, self._buf
        self._buf = ""
        if state == "content":
            if buf:
                content.append(buf)
        elif state == "name":
            pass  # unnamed partial call: drop
        elif state == "value" and self._val_streaming:
            frag = (_json_escape(buf) if buf else "") + '"' + "}"
            self._emit_args(deltas, frag)
            self._args_open = False
        elif state == "value":
            value = _coerce(buf, self._cur_types) \
                if _tool_properties(self._parser._tools, self._name) \
                is not None else buf
            frag = self._arg_prefix() + json.dumps(value, ensure_ascii=False)
            self._args_open = True
            self._emit_args(deltas, frag + "}")
            self._args_open = False
        else:  # key / pre_value / args
            self._emit_args(deltas, "}" if self._args_open else "{}")
            self._args_open = False
        self._state = "content"
        return "".join(content), deltas

    # ── internals ───────────────────────────────────────────────────────

    def _arg_prefix(self) -> str:
        lead = ", " if self._args_open else "{"
        return lead + json.dumps(self._cur_key, ensure_ascii=False) + ": "

    def _emit_args(self, deltas: list[dict[str, Any]], fragment: str) -> None:
        deltas.append({
            "index": self._index,
            "function": {"arguments": fragment},
        })

    def _emit_open(self, deltas: list[dict[str, Any]]) -> None:
        deltas.append({
            "index": self._index,
            "id": f"call_{self._index}",
            "type": "function",
            "function": {"name": self._name, "arguments": ""},
        })
        self._called = True

    def _step(self, content: list[str], deltas: list[dict[str, Any]]) -> bool:
        buf = self._buf

        if self._state == "content":
            idx = buf.find(_TOOL_CALL_START)
            if idx >= 0:
                if buf[:idx]:
                    content.append(buf[:idx])
                self._buf = buf[idx + len(_TOOL_CALL_START):]
                self._index += 1
                self._name = ""
                self._args_open = False
                self._state = "name"
                return True
            hold = _partial_tag_len(buf, _TOOL_CALL_START)
            emit = buf[: len(buf) - hold]
            if emit:
                content.append(emit)
                self._buf = buf[len(buf) - hold:]
            return False

        if self._state == "name":
            k_idx = buf.find(_ARG_KEY_START)
            e_idx = buf.find(_TOOL_CALL_END)
            if k_idx >= 0 and (e_idx < 0 or k_idx < e_idx):
                self._name = buf[:k_idx].strip()
                self._emit_open(deltas)
                self._buf = buf[k_idx + len(_ARG_KEY_START):]
                self._state = "key"
                return True
            if e_idx >= 0:
                self._name = buf[:e_idx].strip()
                self._emit_open(deltas)
                self._emit_args(deltas, "{}")
                self._buf = buf[e_idx + len(_TOOL_CALL_END):]
                self._state = "content"
                return True
            return False  # name incomplete: buffer

        if self._state == "key":
            idx = buf.find(_ARG_KEY_END)
            if idx < 0:
                return False
            self._cur_key = buf[:idx].strip()
            self._buf = buf[idx + len(_ARG_KEY_END):]
            self._state = "pre_value"
            return True

        if self._state == "pre_value":
            idx = buf.find(_ARG_VALUE_START)
            if idx < 0:
                return False
            self._buf = buf[idx + len(_ARG_VALUE_START):]
            props = _tool_properties(self._parser._tools, self._name)
            if props is None:
                # No schema: values stay strings → streamable.
                self._cur_types = ["string"]
            else:
                self._cur_types = _extract_types(props.get(self._cur_key))
            # Only schema-string-only values can stream inside the value:
            # other types may serialize differently once coerced
            # (ref/vllm parser_engine._streamable_string_keys).
            self._val_streaming = set(self._cur_types) == {"string"}
            if self._val_streaming:
                frag = self._arg_prefix() + '"'
                self._args_open = True
                self._emit_args(deltas, frag)
            self._state = "value"
            return True

        if self._state == "value":
            idx = buf.find(_ARG_VALUE_END)
            if self._val_streaming:
                if idx >= 0:
                    chunk = buf[:idx]
                    frag = (_json_escape(chunk) if chunk else "") + '"'
                    self._emit_args(deltas, frag)
                    self._buf = buf[idx + len(_ARG_VALUE_END):]
                    self._state = "args"
                    return True
                hold = _partial_tag_len(buf, _ARG_VALUE_END)
                emit = buf[: len(buf) - hold]
                if emit:
                    self._emit_args(deltas, _json_escape(emit))
                    self._buf = buf[len(buf) - hold:]
                return False
            if idx < 0:
                return False  # buffered value: wait for the close tag
            raw = buf[:idx]
            value = _coerce(raw, self._cur_types)
            frag = self._arg_prefix() + json.dumps(value, ensure_ascii=False)
            self._args_open = True
            self._emit_args(deltas, frag)
            self._buf = buf[idx + len(_ARG_VALUE_END):]
            self._state = "args"
            return True

        # args: between a closed value and the next <arg_key> / </tool_call>
        k_idx = buf.find(_ARG_KEY_START)
        e_idx = buf.find(_TOOL_CALL_END)
        if k_idx >= 0 and (e_idx < 0 or k_idx < e_idx):
            self._buf = buf[k_idx + len(_ARG_KEY_START):]
            self._state = "key"
            return True
        if e_idx >= 0:
            self._emit_args(deltas, "}" if self._args_open else "{}")
            self._args_open = False
            self._buf = buf[e_idx + len(_TOOL_CALL_END):]
            self._state = "content"
            return True
        return False


# ---------------------------------------------------------------------------
# deepseek_v4 — DSML wire format
#
# Ported (lean) from ref/vllm (Apache-2.0, Copyright contributors to the
# vLLM project):
#   - ref/vllm/vllm/parser/deepseek_v4.py            (DSML terminals,
#     _PARAM_RE, _dsml_arg_converter value typing: string="true" → raw
#     string, string="false" → json.loads with raw fallback)
#   - ref/vllm/vllm/tool_parsers/deepseekv4_engine_tool_parser.py
#     (structural_tag_model = "deepseek_v4" — the xgrammar builtin)
#
# DeepSeek-V4 output format (matches the GGUF chat template verbatim):
#
#   <｜DSML｜tool_calls>
#   <｜DSML｜invoke name="func_name">
#   <｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>
#   <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   </｜DSML｜tool_calls>
#
# Unlike glm47, the wire declares each value's type (string="true|false"),
# so no request-schema coercion is needed: string params stay raw text and
# stream as JSON-escaped fragments; non-string params are buffered to the
# close tag and JSON-parsed (raw fallback).  Deviation from the vLLM
# parser (documented): the `_unwrap_wrapper_args` {"arguments"/"input":
# {...}} unwrap nicety is NOT ported — it cannot be applied to
# already-streamed fragments and would break the streaming/non-streaming
# byte-equality contract this registry guarantees.
# ---------------------------------------------------------------------------

_DSML = "｜DSML｜"
_DSML_TOOL_START = f"<{_DSML}tool_calls>"
_DSML_TOOL_END = f"</{_DSML}tool_calls>"
_DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
_DSML_INVOKE_NAME_END = '">'
_DSML_INVOKE_END = f"</{_DSML}invoke>"
_DSML_PARAM_PREFIX = f'<{_DSML}parameter name="'
_DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

_DSML_BLOCK_RE = re.compile(
    re.escape(_DSML_TOOL_START) + r"(?P<body>.*?)" + re.escape(_DSML_TOOL_END),
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    re.escape(_DSML_INVOKE_PREFIX) + r'(?P<name>[^"]+)">'
    r"(?P<body>.*?)" + re.escape(_DSML_INVOKE_END),
    re.DOTALL,
)
# From ref/vllm/vllm/parser/deepseek_v4.py (_PARAM_RE).
_DSML_PARAM_RE = re.compile(
    re.escape(f"<{_DSML}parameter") +
    r'\s+name="(?P<key>[^"]+)"\s+string="(?P<str>true|false)">'
    r"(?P<value>.*?)" + re.escape(_DSML_PARAM_CLOSE),
    re.DOTALL,
)
# Head of one parameter tag AFTER _DSML_PARAM_PREFIX was consumed:
# key + '" string="' + true|false + '">'.
_DSML_PARAM_HEAD_RE = re.compile(
    r'(?P<key>[^"]+)"\s+string="(?P<str>true|false)">')


def _dsml_value(raw: str, is_string: bool) -> Any:
    """DSML value typing (ref/vllm _dsml_arg_converter): the wire's
    string attribute wins — no request-schema coercion."""
    if is_string:
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    return parsed if _json_finite(parsed) else raw


class DeepSeekV4ToolParser:
    """DeepSeek-V4 DSML wire-format tool-call parser (per-request
    instance).  ``tools`` is accepted for registry-signature parity; DSML
    values are self-typed so the schemas are not consulted."""

    name = "deepseek_v4"
    structural_tag_model = "deepseek_v4"

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self._tools = tools

    def extract_tool_calls(self, text: str) -> ParsedToolCalls:
        blocks = list(_DSML_BLOCK_RE.finditer(text))
        if not blocks:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        calls: list[ToolCall] = []
        for block in blocks:
            for m in _DSML_INVOKE_RE.finditer(block.group("body")):
                params: dict[str, Any] = {}
                for pm in _DSML_PARAM_RE.finditer(m.group("body")):
                    params[pm.group("key")] = _dsml_value(
                        pm.group("value"), pm.group("str") == "true")
                calls.append(ToolCall(
                    id=f"call_{len(calls)}",
                    function=ToolCallFunction(
                        name=m.group("name"),
                        arguments=json.dumps(params, ensure_ascii=False),
                    ),
                ))
        if not calls:
            return ParsedToolCalls(tool_calls=(), content=text or None)
        content = text[: blocks[0].start()].strip()
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content or None,
        )

    def stream(self) -> "DeepSeekV4ToolStream":
        return DeepSeekV4ToolStream(self)


class DeepSeekV4ToolStream:
    """Incremental DSML tool-call extractor.

    Same contract as Glm47ToolStream: ``process`` returns
    ``(content_delta, tool_call_deltas)``; per call index the first delta
    opens the call, later deltas carry arguments fragments whose
    concatenation is byte-equal to the non-streaming ``arguments`` JSON;
    ``finish`` closes any open arguments so concatenation stays valid.
    """

    def __init__(self, parser: DeepSeekV4ToolParser) -> None:
        self._parser = parser
        self._buf = ""
        self._state = "content"
        self._index = -1
        self._name = ""
        self._cur_key = ""
        self._val_string = False      # wire string="true" → streamable
        self._args_open = False       # '{' fragment already emitted
        self._called = False

    # ── public API ──────────────────────────────────────────────────────

    @property
    def tools_called(self) -> bool:
        return self._called

    def process(self, delta_text: str) -> tuple[str, list[dict[str, Any]]]:
        self._buf += delta_text
        content: list[str] = []
        deltas: list[dict[str, Any]] = []
        while self._step(content, deltas):
            pass
        return "".join(content), deltas

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        """End-of-stream flush: withheld content is emitted; an open
        call's arguments JSON is closed.  A call cut before its name
        completed never materialized and is dropped."""
        content: list[str] = []
        deltas: list[dict[str, Any]] = []
        state, buf = self._state, self._buf
        self._buf = ""
        if state == "content":
            if buf:
                content.append(buf)
        elif state in ("preamble", "name"):
            pass  # inside the block / unnamed partial call: drop
        elif state == "value" and self._val_string:
            frag = (_json_escape(buf) if buf else "") + '"' + "}"
            self._emit_args(deltas, frag)
            self._args_open = False
        elif state == "value":
            value = _dsml_value(buf, is_string=False)
            frag = self._arg_prefix() + json.dumps(value, ensure_ascii=False)
            self._args_open = True
            self._emit_args(deltas, frag + "}")
            self._args_open = False
        elif state in ("params", "param_head"):
            self._emit_args(deltas, "}" if self._args_open else "{}")
            self._args_open = False
        self._state = "content"
        return "".join(content), deltas

    # ── internals ───────────────────────────────────────────────────────

    def _arg_prefix(self) -> str:
        lead = ", " if self._args_open else "{"
        return lead + json.dumps(self._cur_key, ensure_ascii=False) + ": "

    def _emit_args(self, deltas: list[dict[str, Any]], fragment: str) -> None:
        deltas.append({
            "index": self._index,
            "function": {"arguments": fragment},
        })

    def _emit_open(self, deltas: list[dict[str, Any]]) -> None:
        deltas.append({
            "index": self._index,
            "id": f"call_{self._index}",
            "type": "function",
            "function": {"name": self._name, "arguments": ""},
        })
        self._called = True

    @staticmethod
    def _first_tag(buf: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """(index, tag) of the earliest of ``tags`` in buf; (-1, "") when
        none is present."""
        best_i, best_t = -1, ""
        for t in tags:
            i = buf.find(t)
            if i >= 0 and (best_i < 0 or i < best_i):
                best_i, best_t = i, t
        return best_i, best_t

    def _step(self, content: list[str], deltas: list[dict[str, Any]]) -> bool:
        buf = self._buf

        if self._state == "content":
            idx = buf.find(_DSML_TOOL_START)
            if idx >= 0:
                if buf[:idx]:
                    content.append(buf[:idx])
                self._buf = buf[idx + len(_DSML_TOOL_START):]
                self._state = "preamble"
                return True
            hold = _partial_tag_len(buf, _DSML_TOOL_START)
            emit = buf[: len(buf) - hold]
            if emit:
                content.append(emit)
                self._buf = buf[len(buf) - hold:]
            return False

        if self._state == "preamble":
            # Between <｜DSML｜tool_calls> / </｜DSML｜invoke> and the next
            # invoke (whitespace separators per the template).
            i, tag = self._first_tag(
                buf, (_DSML_INVOKE_PREFIX, _DSML_TOOL_END))
            if i < 0:
                return False
            if tag == _DSML_INVOKE_PREFIX:
                self._buf = buf[i + len(_DSML_INVOKE_PREFIX):]
                self._index += 1
                self._name = ""
                self._args_open = False
                self._state = "name"
                return True
            self._buf = buf[i + len(_DSML_TOOL_END):]
            self._state = "content"
            return True

        if self._state == "name":
            idx = buf.find(_DSML_INVOKE_NAME_END)
            if idx < 0:
                return False
            self._name = buf[:idx]
            self._emit_open(deltas)
            self._buf = buf[idx + len(_DSML_INVOKE_NAME_END):]
            self._state = "params"
            return True

        if self._state == "params":
            i, tag = self._first_tag(
                buf, (_DSML_PARAM_PREFIX, _DSML_INVOKE_END, _DSML_TOOL_END))
            if i < 0:
                return False
            if tag == _DSML_PARAM_PREFIX:
                self._buf = buf[i + len(_DSML_PARAM_PREFIX):]
                self._state = "param_head"
                return True
            # invoke closed (a bare TOOL_END also closes the call —
            # tolerance for truncated model output, ref/vllm transition
            # (TOOL_ARGS, TOOL_END)).
            self._emit_args(deltas, "}" if self._args_open else "{}")
            self._args_open = False
            self._buf = buf[i + len(tag):]
            self._state = "preamble" if tag == _DSML_INVOKE_END else "content"
            return True

        if self._state == "param_head":
            # buf starts right after '<｜DSML｜parameter name="'.
            m = _DSML_PARAM_HEAD_RE.match(buf)
            if m is None:
                if ">" in buf:
                    # Malformed head that can never match: drop the call's
                    # remainder conservatively as content-free noise.
                    self._buf = buf[buf.find(">") + 1:]
                    return True
                return False
            self._cur_key = m.group("key")
            self._val_string = m.group("str") == "true"
            self._buf = buf[m.end():]
            if self._val_string:
                frag = self._arg_prefix() + '"'
                self._args_open = True
                self._emit_args(deltas, frag)
            self._state = "value"
            return True

        # value
        idx = buf.find(_DSML_PARAM_CLOSE)
        if self._val_string:
            if idx >= 0:
                chunk = buf[:idx]
                frag = (_json_escape(chunk) if chunk else "") + '"'
                self._emit_args(deltas, frag)
                self._buf = buf[idx + len(_DSML_PARAM_CLOSE):]
                self._state = "params"
                return True
            hold = _partial_tag_len(buf, _DSML_PARAM_CLOSE)
            emit = buf[: len(buf) - hold]
            if emit:
                self._emit_args(deltas, _json_escape(emit))
                self._buf = buf[len(buf) - hold:]
            return False
        if idx < 0:
            return False  # buffered non-string value: wait for close
        raw = buf[:idx]
        value = _dsml_value(raw, is_string=False)
        frag = self._arg_prefix() + json.dumps(value, ensure_ascii=False)
        self._args_open = True
        self._emit_args(deltas, frag)
        self._buf = buf[idx + len(_DSML_PARAM_CLOSE):]
        self._state = "params"
        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TOOL_PARSERS: dict[str, type] = {
    "glm47": Glm47ToolParser,
    "deepseek_v4": DeepSeekV4ToolParser,
}


def tool_parser_names() -> list[str]:
    return sorted(_TOOL_PARSERS)


def get_tool_parser(name: str) -> type:
    try:
        return _TOOL_PARSERS[name]
    except KeyError:
        raise ValueError(
            f"unknown tool call parser '{name}' "
            f"(available: {', '.join(tool_parser_names())})",
        ) from None
