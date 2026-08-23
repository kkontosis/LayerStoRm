"""Tests for the named tool-call parser layer (server.tool_parsers).

Non-streaming cases mirror ref/vllm/tests/tool_parsers/
test_glm47_moe_tool_parser.py; streaming cases feed the parser
chunk-by-chunk and assert the OpenAI delta sequence.
"""

from __future__ import annotations

import json

import pytest

from server.tool_parsers import (
    Glm47ToolParser,
    get_tool_parser,
    tool_parser_names,
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_current_date", "parameters": {}},
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "detailed": {"type": "boolean"},
                    "coords": {"type": "object"},
                    "temps": {"type": "array"},
                    "factor": {"type": "number"},
                },
            },
        },
    },
]


def _parser(tools=SAMPLE_TOOLS) -> Glm47ToolParser:
    return Glm47ToolParser(tools=tools)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_get_glm47(self):
        assert get_tool_parser("glm47") is Glm47ToolParser

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown tool call parser"):
            get_tool_parser("nope")

    def test_names(self):
        assert "glm47" in tool_parser_names()


# ---------------------------------------------------------------------------
# Non-streaming extraction
# ---------------------------------------------------------------------------

class TestExtractToolCalls:

    def test_no_tool_call(self):
        r = _parser().extract_tool_calls("This is a plain response.")
        assert not r.tool_calls
        assert r.content == "This is a plain response."

    def test_zero_arg_inline(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_current_date</tool_call>")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_current_date\n</tool_call>")
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Beijing</arg_value></tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "Beijing"}

    def test_args_with_newlines(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
            "<arg_value>Beijing</arg_value>\n</tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>  Beijing  </arg_value></tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "  Beijing  "}

    def test_content_before(self):
        r = _parser().extract_tool_calls(
            "Checking.<tool_call>get_current_date</tool_call>")
        assert r.tool_calls
        assert r.content == "Checking."

    def test_whitespace_content_none(self):
        r = _parser().extract_tool_calls(
            "  \n  <tool_call>get_current_date</tool_call>")
        assert r.content is None

    def test_multiple(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Shanghai</arg_value></tool_call>")
        assert len(r.tool_calls) == 2
        assert r.tool_calls[0].id == "call_0"
        assert r.tool_calls[1].id == "call_1"
        assert json.loads(r.tool_calls[1].function.arguments) == {
            "city": "Shanghai"}

    def test_schema_type_coercion(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "<arg_key>detailed</arg_key><arg_value>true</arg_value>"
            "<arg_key>factor</arg_key><arg_value>2.5</arg_value>"
            "</tool_call>")
        args = json.loads(r.tool_calls[0].function.arguments)
        assert args == {
            "city": "Paris", "days": 3, "detailed": True, "factor": 2.5}

    def test_nested_json_argument(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather"
            '<arg_key>coords</arg_key><arg_value>{"lat": 48.8, '
            '"nested": {"a": [1, 2]}}</arg_value>'
            "<arg_key>temps</arg_key><arg_value>[1, 2, 3]</arg_value>"
            "</tool_call>")
        args = json.loads(r.tool_calls[0].function.arguments)
        assert args["coords"] == {"lat": 48.8, "nested": {"a": [1, 2]}}
        assert args["temps"] == [1, 2, 3]

    def test_numeric_string_stays_string_with_string_schema(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>123</arg_value></tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "123"}

    def test_no_schema_values_stay_strings(self):
        r = Glm47ToolParser(tools=None).extract_tool_calls(
            "<tool_call>anything<arg_key>x</arg_key>"
            "<arg_value>5</arg_value></tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {"x": "5"}

    def test_unknown_key_defaults_to_string(self):
        r = _parser().extract_tool_calls(
            "<tool_call>get_weather<arg_key>mystery</arg_key>"
            "<arg_value>42</arg_value></tool_call>")
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "mystery": "42"}

    def test_malformed_unterminated_is_content(self):
        text = "<tool_call>get_weather<arg_key>city</arg_key>"
        r = _parser().extract_tool_calls(text)
        assert not r.tool_calls
        assert r.content == text


# ---------------------------------------------------------------------------
# Streaming extraction
# ---------------------------------------------------------------------------

def _drive(chunks, tools=SAMPLE_TOOLS):
    """Feed chunks through a stream; returns (content, deltas)."""
    stream = Glm47ToolParser(tools=tools).stream()
    content = []
    deltas = []
    for chunk in chunks:
        c, d = stream.process(chunk)
        if c:
            content.append(c)
        deltas.extend(d)
    c, d = stream.finish()
    if c:
        content.append(c)
    deltas.extend(d)
    return "".join(content), deltas, stream


def _join_args(deltas, index=0):
    return "".join(
        d["function"]["arguments"] for d in deltas
        if d["index"] == index and "arguments" in d.get("function", {})
    )


def _names(deltas):
    return [d["function"]["name"] for d in deltas
            if d.get("function", {}).get("name")]


class TestStreaming:

    def test_no_args(self):
        content, deltas, stream = _drive(
            ["<tool_call>", "get_current_date", "</tool_call>"])
        assert content == ""
        assert _names(deltas) == ["get_current_date"]
        assert deltas[0]["id"] == "call_0"
        assert deltas[0]["index"] == 0
        assert deltas[0]["type"] == "function"
        assert json.loads(_join_args(deltas)) == {}
        assert stream.tools_called

    def test_with_args_token_chunks(self):
        content, deltas, stream = _drive([
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Bei",
            "jing",
            "</arg_value>",
            "</tool_call>",
        ])
        assert content == ""
        assert _names(deltas) == ["get_weather"]
        assert json.loads(_join_args(deltas)) == {"city": "Beijing"}
        # String-typed value streams incrementally: at least two argument
        # fragments carry pieces of the value.
        frags = [d["function"]["arguments"] for d in deltas
                 if "arguments" in d.get("function", {})]
        assert any("Bei" in f and "jing" not in f for f in frags)

    def test_matches_non_streaming_json(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "</tool_call>"
        )
        _, deltas, _ = _drive([text])
        non_stream = _parser().extract_tool_calls(text)
        assert _join_args(deltas) == non_stream.tool_calls[0].function.arguments

    def test_non_string_value_buffered(self):
        content, deltas, _ = _drive([
            "<tool_call>get_weather<arg_key>days</arg_key><arg_value>",
            "1",
            "2",
            "</arg_value></tool_call>",
        ])
        assert json.loads(_join_args(deltas)) == {"days": 12}

    def test_content_around_calls(self):
        content, deltas, _ = _drive([
            "Let me check.",
            "<tool_call>get_current_date</tool_call>",
            " done",
        ])
        assert content == "Let me check. done"
        assert _names(deltas) == ["get_current_date"]

    def test_partial_tag_withheld(self):
        content, deltas, _ = _drive([
            "hello <tool_",
            "call>get_current_date</tool_call>",
        ])
        assert content == "hello "
        assert _names(deltas) == ["get_current_date"]

    def test_partial_tag_flushed_as_content(self):
        content, deltas, _ = _drive(["text ends with <tool_c"])
        assert content == "text ends with <tool_c"
        assert deltas == []

    def test_multiple_calls_indices(self):
        _, deltas, _ = _drive([
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Shanghai</arg_value></tool_call>",
        ])
        assert _names(deltas) == ["get_weather", "get_weather"]
        opens = [d for d in deltas if "id" in d]
        assert [d["index"] for d in opens] == [0, 1]
        assert [d["id"] for d in opens] == ["call_0", "call_1"]
        assert json.loads(_join_args(deltas, 0)) == {"city": "Beijing"}
        assert json.loads(_join_args(deltas, 1)) == {"city": "Shanghai"}

    def test_finish_mid_string_value_closes_json(self):
        _, deltas, _ = _drive([
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
        ])
        assert json.loads(_join_args(deltas)) == {"city": "Par"}

    def test_finish_mid_buffered_value_closes_json(self):
        _, deltas, _ = _drive([
            "<tool_call>get_weather<arg_key>days</arg_key><arg_value>7",
        ])
        assert json.loads(_join_args(deltas)) == {"days": 7}

    def test_finish_mid_name_drops_call(self):
        content, deltas, stream = _drive(["<tool_call>get_wea"])
        assert deltas == []
        assert not stream.tools_called

    def test_value_with_json_escapes(self):
        _, deltas, _ = _drive([
            "<tool_call>get_weather<arg_key>city</arg_key>"
            '<arg_value>a "quoted"\nline</arg_value></tool_call>',
        ])
        assert json.loads(_join_args(deltas)) == {"city": 'a "quoted"\nline'}

    def test_no_call_plain_content(self):
        content, deltas, stream = _drive(["Hello ", "world."])
        assert content == "Hello world."
        assert deltas == []
        assert not stream.tools_called
