"""Tests for the deepseek_v4 serving parsers (ticket I).

DSML tool-call parser (server.tool_parsers.DeepSeekV4ToolParser/Stream):
non-streaming cases mirror ref/vllm/tests/tool_parsers/
test_deepseekv4_tool_parser.py shapes; streaming cases feed scripted
token-sized chunks and assert the OpenAI delta sequence, with the
concatenated arguments fragments byte-equal to the non-streaming JSON.

Reasoning parser (server.reasoning_parsers.DeepSeekV4ReasoningParser):
STRUCTURAL start boundary — the V4 template pre-opens <think>, so model
output begins inside reasoning with no opening marker; only </think>
closes it.  --reasoning-config marker overrides force the text path.
"""

from __future__ import annotations

import json

import pytest

from server.reasoning_parsers import (
    DeepSeekV4ReasoningParser,
    Glm45ReasoningParser,
    get_reasoning_parser,
)
from server.tool_parsers import (
    DeepSeekV4ToolParser,
    Glm47ToolParser,
    get_tool_parser,
    tool_parser_names,
)

DSML = "｜DSML｜"
TOOL_START = f"<{DSML}tool_calls>"
TOOL_END = f"</{DSML}tool_calls>"


def _invoke(name: str, params: list[tuple[str, str, str]]) -> str:
    """One <｜DSML｜invoke> block; params = (key, string_attr, value)."""
    out = [f'<{DSML}invoke name="{name}">\n']
    for key, s, val in params:
        out.append(
            f'<{DSML}parameter name="{key}" string="{s}">{val}'
            f"</{DSML}parameter>\n")
    out.append(f"</{DSML}invoke>\n")
    return "".join(out)


def _wire(*invokes: str) -> str:
    return TOOL_START + "\n" + "".join(invokes) + TOOL_END


WEATHER_CALL = _wire(_invoke("get_weather", [
    ("city", "true", "Paris"),
    ("days", "false", "5"),
]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_get_deepseek_v4(self):
        assert get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser

    def test_names(self):
        assert "deepseek_v4" in tool_parser_names()
        assert "glm47" in tool_parser_names()

    def test_structural_tag_models(self):
        assert DeepSeekV4ToolParser.structural_tag_model == "deepseek_v4"
        assert Glm47ToolParser.structural_tag_model == "glm_4_7"

    def test_reasoning_registry(self):
        assert get_reasoning_parser("deepseek_v4") is DeepSeekV4ReasoningParser


# ---------------------------------------------------------------------------
# Non-streaming DSML extraction
# ---------------------------------------------------------------------------

class TestExtract:

    def test_no_tool_call(self):
        parsed = DeepSeekV4ToolParser().extract_tool_calls("hello world")
        assert parsed.tool_calls == ()
        assert parsed.content == "hello world"

    def test_string_and_number_params(self):
        parsed = DeepSeekV4ToolParser().extract_tool_calls(WEATHER_CALL)
        assert len(parsed.tool_calls) == 1
        tc = parsed.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {
            "city": "Paris", "days": 5}
        assert parsed.content is None

    def test_string_param_stays_raw(self):
        # string="true" wins over any JSON-looking content.
        wire = _wire(_invoke("f", [("q", "true", '{"not": "parsed"}')]))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert json.loads(parsed.tool_calls[0].function.arguments) == {
            "q": '{"not": "parsed"}'}

    def test_non_string_json_types(self):
        wire = _wire(_invoke("f", [
            ("flag", "false", "true"),
            ("arr", "false", "[1, 2, 3]"),
            ("obj", "false", '{"a": 1}'),
            ("null_v", "false", "null"),
        ]))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert json.loads(parsed.tool_calls[0].function.arguments) == {
            "flag": True, "arr": [1, 2, 3], "obj": {"a": 1}, "null_v": None}

    def test_non_string_bad_json_falls_back_raw(self):
        wire = _wire(_invoke("f", [("x", "false", "not json")]))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert json.loads(parsed.tool_calls[0].function.arguments) == {
            "x": "not json"}

    def test_zero_param_call(self):
        wire = _wire(_invoke("get_current_date", []))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert parsed.tool_calls[0].function.name == "get_current_date"
        assert parsed.tool_calls[0].function.arguments == "{}"

    def test_parallel_invokes(self):
        wire = _wire(
            _invoke("a", [("x", "false", "1")]),
            _invoke("b", [("y", "true", "two")]),
        )
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert [t.function.name for t in parsed.tool_calls] == ["a", "b"]
        assert [t.id for t in parsed.tool_calls] == ["call_0", "call_1"]
        assert json.loads(parsed.tool_calls[1].function.arguments) == {
            "y": "two"}

    def test_content_before_call(self):
        text = "Let me check.\n\n" + WEATHER_CALL
        parsed = DeepSeekV4ToolParser().extract_tool_calls(text)
        assert parsed.content == "Let me check."
        assert len(parsed.tool_calls) == 1

    def test_unterminated_block_is_content(self):
        text = TOOL_START + '\n<' + DSML + 'invoke name="f">'
        parsed = DeepSeekV4ToolParser().extract_tool_calls(text)
        assert parsed.tool_calls == ()
        assert parsed.content == text

    def test_multiline_string_value_preserved(self):
        wire = _wire(_invoke("f", [("text", "true", "line1\n  line2\t")]))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert json.loads(parsed.tool_calls[0].function.arguments) == {
            "text": "line1\n  line2\t"}

    def test_unicode_value(self):
        wire = _wire(_invoke("f", [("city", "true", "杭州")]))
        parsed = DeepSeekV4ToolParser().extract_tool_calls(wire)
        assert json.loads(parsed.tool_calls[0].function.arguments) == {
            "city": "杭州"}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _drive(chunks):
    stream = DeepSeekV4ToolParser().stream()
    content = []
    deltas = []
    for ch in chunks:
        c, d = stream.process(ch)
        content.append(c)
        deltas.extend(d)
    c, d = stream.finish()
    content.append(c)
    deltas.extend(d)
    return "".join(content), deltas, stream


def _chunked(text: str, n: int = 4) -> list[str]:
    return [text[i:i + n] for i in range(0, len(text), n)]


def _join_args(deltas, index=0):
    return "".join(
        d["function"]["arguments"] for d in deltas
        if d["index"] == index and "arguments" in d.get("function", {}))


def _names(deltas):
    return [d["function"]["name"] for d in deltas
            if "name" in d.get("function", {})]


class TestStreaming:

    def test_matches_non_streaming_json(self):
        non_stream = DeepSeekV4ToolParser().extract_tool_calls(WEATHER_CALL)
        for n in (1, 3, 7, len(WEATHER_CALL)):
            content, deltas, stream = _drive(_chunked(WEATHER_CALL, n))
            assert stream.tools_called
            assert _names(deltas) == ["get_weather"]
            assert _join_args(deltas) == \
                non_stream.tool_calls[0].function.arguments, f"chunk={n}"

    def test_zero_param(self):
        wire = _wire(_invoke("get_current_date", []))
        content, deltas, stream = _drive(_chunked(wire))
        assert _names(deltas) == ["get_current_date"]
        assert _join_args(deltas) == "{}"

    def test_parallel_invokes_indices(self):
        wire = _wire(
            _invoke("a", [("x", "false", "1")]),
            _invoke("b", [("y", "true", "two")]),
        )
        non_stream = DeepSeekV4ToolParser().extract_tool_calls(wire)
        content, deltas, stream = _drive(_chunked(wire, 5))
        assert _names(deltas) == ["a", "b"]
        for i in (0, 1):
            assert _join_args(deltas, i) == \
                non_stream.tool_calls[i].function.arguments

    def test_content_before_and_partial_tag_withheld(self):
        text = "thinking done" + WEATHER_CALL
        content, deltas, _ = _drive(_chunked(text, 3))
        assert content == "thinking done"
        assert _names(deltas) == ["get_weather"]

    def test_partial_tag_flushed_as_content(self):
        # A prefix of the DSML start tag at end-of-stream is plain text.
        content, deltas, _ = _drive(["see <", DSML[:2]])
        assert content == "see <" + DSML[:2]
        assert deltas == []

    def test_finish_mid_string_value_closes_json(self):
        head = (TOOL_START + "\n" + f'<{DSML}invoke name="f">\n'
                + f'<{DSML}parameter name="q" string="true">par')
        content, deltas, _ = _drive(_chunked(head, 6))
        args = _join_args(deltas)
        assert json.loads(args) == {"q": "par"}

    def test_finish_mid_buffered_value_closes_json(self):
        head = (TOOL_START + "\n" + f'<{DSML}invoke name="f">\n'
                + f'<{DSML}parameter name="n" string="false">42')
        content, deltas, _ = _drive(_chunked(head, 6))
        assert json.loads(_join_args(deltas)) == {"n": 42}

    def test_finish_mid_name_drops_call(self):
        head = TOOL_START + "\n" + f'<{DSML}invoke name="ge'
        content, deltas, stream = _drive(_chunked(head, 5))
        assert deltas == []
        assert not stream.tools_called

    def test_truncated_after_params_tool_end_closes(self):
        # Model closed the block without </invoke> (ref/vllm
        # (TOOL_ARGS, TOOL_END) transition tolerance).
        wire = (TOOL_START + "\n" + f'<{DSML}invoke name="f">\n'
                + f'<{DSML}parameter name="q" string="true">v'
                + f"</{DSML}parameter>\n" + TOOL_END)
        content, deltas, _ = _drive(_chunked(wire, 7))
        assert json.loads(_join_args(deltas)) == {"q": "v"}

    def test_value_with_json_escapes(self):
        wire = _wire(_invoke("f", [("q", "true", 'say "hi"\\n')]))
        non_stream = DeepSeekV4ToolParser().extract_tool_calls(wire)
        content, deltas, _ = _drive(_chunked(wire, 2))
        assert _join_args(deltas) == \
            non_stream.tool_calls[0].function.arguments

    def test_unicode_streaming_byte_equal(self):
        wire = _wire(_invoke("f", [("city", "true", "杭州市")]))
        non_stream = DeepSeekV4ToolParser().extract_tool_calls(wire)
        content, deltas, _ = _drive(list(wire))  # 1-char chunks
        assert _join_args(deltas) == \
            non_stream.tool_calls[0].function.arguments


# ---------------------------------------------------------------------------
# deepseek_v4 reasoning parser — structural start boundary
# ---------------------------------------------------------------------------

THINK_START_ID = 128821
THINK_END_ID = 128822


def _rparser() -> DeepSeekV4ReasoningParser:
    return DeepSeekV4ReasoningParser(THINK_START_ID, THINK_END_ID)


class TestReasoningStructural:

    def test_markers(self):
        assert DeepSeekV4ReasoningParser.start_str == ""
        assert DeepSeekV4ReasoningParser.end_str == "</think>"

    def test_extract_no_opening_marker(self):
        # Output begins in reasoning (template pre-opened <think>).
        reasoning, content = _rparser().extract_reasoning(
            "figuring...</think>The answer is 4.")
        assert reasoning == "figuring..."
        assert content == "The answer is 4."

    def test_extract_all_reasoning_when_unclosed(self):
        reasoning, content = _rparser().extract_reasoning("still thinking")
        assert reasoning == "still thinking"
        assert content is None

    def test_split_token_ids(self):
        ids = [10, 11, THINK_END_ID, 20, 21]
        reasoning_ids, content_ids = _rparser().split_token_ids(ids)
        assert reasoning_ids == [10, 11]
        assert content_ids == [20, 21]

    def test_split_token_ids_spurious_start_absorbed(self):
        ids = [THINK_START_ID, 10, THINK_END_ID, 20]
        reasoning_ids, content_ids = _rparser().split_token_ids(ids)
        assert reasoning_ids == [10]
        assert content_ids == [20]

    def test_stream_token_mode(self):
        s = _rparser().stream(thinking=True)
        assert s.process("reason", [10]) == ("reason", "")
        assert s.process("</think>ans", [THINK_END_ID, 20]) == ("", "ans")
        assert s.process(" more", [21]) == ("", " more")

    def test_stream_thinking_disabled_all_content(self):
        s = _rparser().stream(thinking=False)
        assert s.process("plain", [10]) == ("", "plain")

    def test_stream_text_fallback_marker_split(self):
        p = DeepSeekV4ReasoningParser()          # no token ids
        s = p.stream(thinking=True)
        r1, c1 = s.process("abc</th")
        r2, c2 = s.process("ink>def")
        assert (r1 + r2, c1 + c2) == ("abc", "def")


class TestReasoningConfigOverrides:

    def test_marker_override_forces_text_path(self):
        p = Glm45ReasoningParser(
            90, 91, reasoning_start_str="<reason>",
            reasoning_end_str="</reason>")
        assert p.start_str == "<reason>"
        assert p.end_str == "</reason>"
        assert not p.has_token_ids
        assert p.split_token_ids([1, 91, 2]) is None
        reasoning, content = p.extract_reasoning("<reason>a</reason>b")
        assert (reasoning, content) == ("a", "b")

    def test_empty_marker_override_structural(self):
        p = Glm45ReasoningParser(
            90, 91, reasoning_start_str="", reasoning_end_str="</think>")
        reasoning, content = p.extract_reasoning("a</think>b")
        assert (reasoning, content) == ("a", "b")

    def test_class_markers_unaffected(self):
        Glm45ReasoningParser(90, 91, reasoning_start_str="<x>")
        assert Glm45ReasoningParser.start_str == "<think>"

    def test_no_override_keeps_token_ids(self):
        p = DeepSeekV4ReasoningParser(THINK_START_ID, THINK_END_ID)
        assert p.has_token_ids
