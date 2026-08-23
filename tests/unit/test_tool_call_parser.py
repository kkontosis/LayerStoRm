"""Tests for tool call parser."""

from __future__ import annotations

import json

import pytest

from tokenizer.tool_call_parser import (
    DeepSeekToolCallParser,
    GenericToolCallParser,
    GlmToolCallParser,
    KimiToolCallParser,
    ParsedToolCalls,
    ToolCall,
    ToolCallFunction,
    get_tool_call_parser,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_tool_call_function_frozen(self):
        f = ToolCallFunction(name="get_weather", arguments='{"city":"NYC"}')
        assert f.name == "get_weather"
        with pytest.raises(AttributeError):
            f.name = "x"  # type: ignore[misc]

    def test_tool_call_defaults(self):
        tc = ToolCall(
            id="call_0",
            function=ToolCallFunction(name="f", arguments="{}"),
        )
        assert tc.type == "function"

    def test_parsed_tool_calls(self):
        p = ParsedToolCalls(tool_calls=(), content=None)
        assert len(p.tool_calls) == 0


# ---------------------------------------------------------------------------
# DeepSeek V3
# ---------------------------------------------------------------------------

class TestDeepSeekParser:

    parser = DeepSeekToolCallParser()

    def test_single_tool(self):
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"city": "NYC"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_0"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "NYC"}
        assert result.content is None

    def test_parallel_tools(self):
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"city": "NYC"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search_hotels\n"
            "```json\n"
            '{"location": "NYC"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "search_hotels"
        assert result.tool_calls[0].id == "call_0"
        assert result.tool_calls[1].id == "call_1"

    def test_content_before_tools(self):
        text = (
            "Let me check the weather for you.\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"city": "NYC"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.parse(text)
        assert result.content == "Let me check the weather for you."
        assert len(result.tool_calls) == 1

    def test_no_tools_passthrough(self):
        text = "Just a regular response with no tool calls."
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 0
        assert result.content == text

    def test_malformed_json_graceful(self):
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>broken\n"
            "```json\n"
            "{invalid json\n"
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.arguments == "{invalid json"

    def test_nested_json_args(self):
        args = json.dumps({"query": {"filter": {"type": "weather"}, "limit": 10}})
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
            "```json\n"
            f"{args}\n"
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": {"filter": {"type": "weather"}, "limit": 10},
        }

    def test_empty_text(self):
        result = self.parser.parse("")
        assert result.content is None
        assert len(result.tool_calls) == 0


# ---------------------------------------------------------------------------
# Kimi K2.5
# ---------------------------------------------------------------------------

class TestKimiParser:

    parser = KimiToolCallParser()

    def test_single_tool(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            "<|tool_call_argument_begin|>"
            '{"city": "NYC"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "functions.get_weather:0"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "NYC"}

    def test_parallel_tools(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            "<|tool_call_argument_begin|>"
            '{"city": "NYC"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.search_hotels:1"
            "<|tool_call_argument_begin|>"
            '{"location": "NYC"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "search_hotels"
        assert result.tool_calls[1].id == "functions.search_hotels:1"

    def test_content_before_tools(self):
        text = (
            "I'll look that up."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:0"
            "<|tool_call_argument_begin|>"
            '{"q": "test"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.parser.parse(text)
        assert result.content == "I'll look that up."
        assert len(result.tool_calls) == 1

    def test_no_tools_passthrough(self):
        text = "No tools here."
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 0
        assert result.content == text

    def test_name_extraction_from_id(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>get_weather:0"
            "<|tool_call_argument_begin|>{}<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.parser.parse(text)
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].id == "get_weather:0"

    def test_malformed_no_matches(self):
        text = (
            "<|tool_calls_section_begin|>"
            "garbage content without proper markers"
            "<|tool_calls_section_end|>"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 0
        assert result.content == text


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class TestGenericParser:

    parser = GenericToolCallParser()

    def test_json_array_in_code_block(self):
        text = (
            "Here are the results:\n"
            "```json\n"
            '[{"name": "get_weather", "arguments": {"city": "NYC"}}]\n'
            "```"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].id == "call_0"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "NYC"}

    def test_single_object_in_code_block(self):
        text = (
            "```json\n"
            '{"name": "search", "arguments": {"q": "test"}}\n'
            "```"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_string_arguments(self):
        text = (
            '```json\n'
            '{"name": "f", "arguments": "{\\"a\\": 1}"}\n'
            '```'
        )
        result = self.parser.parse(text)
        assert result.tool_calls[0].function.arguments == '{"a": 1}'

    def test_no_code_block_passthrough(self):
        text = "Just a regular response."
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 0
        assert result.content == text

    def test_malformed_json_in_block(self):
        text = "```json\n{bad json}\n```"
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 0
        assert result.content == text

    def test_content_extracted(self):
        text = (
            "Let me help.\n"
            "```json\n"
            '[{"name": "f", "arguments": {}}]\n'
            "```\n"
            "Done."
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert "Let me help." in result.content
        assert "Done." in result.content

    def test_code_block_without_json_tag(self):
        text = (
            "```\n"
            '[{"name": "f", "arguments": {}}]\n'
            "```"
        )
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestGlmParser:
    """GLM-5/GLM-5.2 <tool_call>{name}<arg_key>/<arg_value> format."""

    def test_no_tool_calls(self):
        r = GlmToolCallParser().parse("plain text answer")
        assert r.tool_calls == ()
        assert r.content == "plain text answer"

    def test_single_call_typed_args(self):
        text = ("Let me check.<tool_call>get_weather"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "<arg_key>days</arg_key><arg_value>3</arg_value>"
                "<arg_key>metric</arg_key><arg_value>true</arg_value>"
                "</tool_call>")
        r = GlmToolCallParser().parse(text)
        assert len(r.tool_calls) == 1
        assert r.content == "Let me check."
        tc = r.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {
            "city": "Paris", "days": 3, "metric": True}

    def test_multiple_calls_and_json_object_arg(self):
        text = ("<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
                "<tool_call>b<arg_key>cfg</arg_key>"
                "<arg_value>{\"k\": [1, 2]}</arg_value></tool_call>")
        r = GlmToolCallParser().parse(text)
        assert [c.function.name for c in r.tool_calls] == ["a", "b"]
        assert json.loads(r.tool_calls[1].function.arguments) == {"cfg": {"k": [1, 2]}}
        assert r.content is None

    def test_no_args_call(self):
        r = GlmToolCallParser().parse("<tool_call>ping</tool_call>")
        assert r.tool_calls[0].function.name == "ping"
        assert json.loads(r.tool_calls[0].function.arguments) == {}


class TestRegistry:

    def test_deepseek_v32(self):
        parser = get_tool_call_parser("deepseek_v32")
        assert isinstance(parser, DeepSeekToolCallParser)

    def test_deepseek_v3(self):
        parser = get_tool_call_parser("deepseek_v3")
        assert isinstance(parser, DeepSeekToolCallParser)

    def test_kimi_k25(self):
        parser = get_tool_call_parser("kimi_k25")
        assert isinstance(parser, KimiToolCallParser)

    def test_unknown_falls_back_to_generic(self):
        parser = get_tool_call_parser("unknown_model")
        assert isinstance(parser, GenericToolCallParser)

    def test_glm_moe_dsa(self):
        parser = get_tool_call_parser("glm_moe_dsa")
        assert isinstance(parser, GlmToolCallParser)
