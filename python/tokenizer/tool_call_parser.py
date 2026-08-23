"""Tool call parser — extracts structured tool calls from model output.

Parses model-specific tool call formats into OpenAI-compatible tool_calls
response objects. Per-model parser selected by ``model_type`` from config.json.
Wire-format handling follows the corresponding vLLM tool parsers (ref/vllm;
Apache-2.0, Copyright contributors to the vLLM project — see
THIRD_PARTY_NOTICES.md).

Supported formats:
  DeepSeek V3 — fullwidth unicode delimiters with JSON in markdown
  Kimi K2.5   — ASCII section markers with ID-based tool calls
  GLM-5       — <tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
  Generic     — JSON arrays/objects in markdown code blocks
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCallFunction:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: ToolCallFunction
    type: str = "function"


@dataclass(frozen=True)
class ParsedToolCalls:
    tool_calls: tuple[ToolCall, ...]
    content: str | None


class ToolCallParser(Protocol):
    def parse(self, text: str) -> ParsedToolCalls: ...


# ---------------------------------------------------------------------------
# DeepSeek V3
# ---------------------------------------------------------------------------

_DS_SECTION_START = "<｜tool▁calls▁begin｜>"

_DS_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<name>.*?)\n"
    r"```json\n(?P<args>.*?)\n```\n?"
    r"<｜tool▁call▁end｜>",
    re.DOTALL,
)


class DeepSeekToolCallParser:

    def parse(self, text: str) -> ParsedToolCalls:
        if _DS_SECTION_START not in text:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        content_part = text[: text.index(_DS_SECTION_START)]
        matches = _DS_CALL_RE.findall(text)
        if not matches:
            return ParsedToolCalls(tool_calls=(), content=text)

        calls: list[ToolCall] = []
        for i, (_, name, args) in enumerate(matches):
            args_str = args.strip()
            if not _is_valid_json(args_str):
                args_str = args.strip()
            calls.append(ToolCall(
                id=f"call_{i}",
                function=ToolCallFunction(
                    name=name.strip(),
                    arguments=args_str,
                ),
            ))
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content_part.strip() or None,
        )


# ---------------------------------------------------------------------------
# Kimi K2.5
# ---------------------------------------------------------------------------

_KIMI_SECTION_START = "<|tool_calls_section_begin|>"

_KIMI_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<id>[^<]+?)\s*"
    r"<\|tool_call_argument_begin\|>\s*"
    r"(?P<args>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)


class KimiToolCallParser:

    def parse(self, text: str) -> ParsedToolCalls:
        if _KIMI_SECTION_START not in text:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        content_part = text[: text.index(_KIMI_SECTION_START)]
        matches = _KIMI_CALL_RE.findall(text)
        if not matches:
            return ParsedToolCalls(tool_calls=(), content=text)

        calls: list[ToolCall] = []
        for call_id, args in matches:
            call_id = call_id.strip()
            name = call_id.split(":")[0].split(".")[-1] if call_id else ""
            calls.append(ToolCall(
                id=call_id,
                function=ToolCallFunction(
                    name=name,
                    arguments=args.strip(),
                ),
            ))
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content_part.strip() or None,
        )


# ---------------------------------------------------------------------------
# GLM-5 / GLM-5.2 (glm_moe_dsa)
# ---------------------------------------------------------------------------
#
# Template emits:  <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# (chat_template.jinja lines 35/84-85). Values are raw strings when the
# argument was a string, JSON otherwise — mirror that on parse: try JSON,
# fall back to the raw string.

_GLM_CALL_RE = re.compile(
    r"<tool_call>(?P<name>[^<]*?)"
    r"(?P<args>(?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)*)"
    r"</tool_call>",
    re.DOTALL,
)

_GLM_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key><arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)


class GlmToolCallParser:

    def parse(self, text: str) -> ParsedToolCalls:
        matches = list(_GLM_CALL_RE.finditer(text))
        if not matches:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        calls: list[ToolCall] = []
        for i, m in enumerate(matches):
            args: dict[str, object] = {}
            for am in _GLM_ARG_RE.finditer(m.group("args")):
                raw = am.group("value")
                # The template tojson-encodes non-string values (numbers,
                # bools, dicts, lists) and emits strings raw — so decode JSON
                # when it parses to a non-string, else keep the raw string.
                # (A raw string that LOOKS like JSON, e.g. "5", is inherently
                # ambiguous on this wire format; prefer the typed reading.)
                try:
                    parsed = json.loads(raw)
                    args[am.group("key")] = raw if isinstance(parsed, str) else parsed
                except (json.JSONDecodeError, ValueError):
                    args[am.group("key")] = raw
            calls.append(ToolCall(
                id=f"call_{i}",
                function=ToolCallFunction(
                    name=m.group("name").strip(),
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            ))
        content_part = text[: matches[0].start()]
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content_part.strip() or None,
        )


# ---------------------------------------------------------------------------
# Generic (JSON in markdown code blocks)
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


class GenericToolCallParser:

    def parse(self, text: str) -> ParsedToolCalls:
        blocks = _CODE_BLOCK_RE.findall(text)
        if not blocks:
            return ParsedToolCalls(tool_calls=(), content=text or None)

        calls: list[ToolCall] = []
        for block in blocks:
            parsed = _try_parse_json(block.strip())
            if parsed is None:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            for i, item in enumerate(items):
                if not isinstance(item, dict) or "name" not in item:
                    continue
                raw_args = item.get("arguments", {})
                args_str = (
                    raw_args if isinstance(raw_args, str)
                    else json.dumps(raw_args, ensure_ascii=False)
                )
                calls.append(ToolCall(
                    id=f"call_{len(calls)}",
                    function=ToolCallFunction(
                        name=item["name"],
                        arguments=args_str,
                    ),
                ))

        if not calls:
            return ParsedToolCalls(tool_calls=(), content=text)

        content_part = _CODE_BLOCK_RE.sub("", text).strip()
        return ParsedToolCalls(
            tool_calls=tuple(calls),
            content=content_part or None,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSER_REGISTRY: dict[str, type[ToolCallParser]] = {
    "deepseek_v32": DeepSeekToolCallParser,
    "deepseek_v3": DeepSeekToolCallParser,
    "kimi_k25": KimiToolCallParser,
    "glm_moe_dsa": GlmToolCallParser,  # GLM-5 / GLM-5.2
}


def get_tool_call_parser(model_type: str) -> ToolCallParser:
    cls = _PARSER_REGISTRY.get(model_type, GenericToolCallParser)
    return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _try_parse_json(s: str) -> dict | list | None:
    try:
        result = json.loads(s)
        if isinstance(result, (dict, list)):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None
