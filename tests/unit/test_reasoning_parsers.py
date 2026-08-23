"""Tests for the reasoning parser layer (server.reasoning_parsers)."""

from __future__ import annotations

import pytest

from server.reasoning_parsers import (
    Glm45ReasoningParser,
    get_reasoning_parser,
    reasoning_parser_names,
)

THINK_START = 90
THINK_END = 91


def _parser() -> Glm45ReasoningParser:
    return Glm45ReasoningParser(THINK_START, THINK_END)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_glm45(self):
        assert get_reasoning_parser("glm45") is Glm45ReasoningParser

    def test_glm47_alias(self):
        assert get_reasoning_parser("glm47") is Glm45ReasoningParser

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown reasoning parser"):
            get_reasoning_parser("nope")

    def test_names(self):
        assert {"glm45", "glm47"} <= set(reasoning_parser_names())


# ---------------------------------------------------------------------------
# Non-streaming: token-id split
# ---------------------------------------------------------------------------

class TestSplitTokenIds:

    def test_reasoning_and_content(self):
        # (reasoning tokens) </think> (content tokens)
        ids = [1, 2, THINK_END, 3, 4]
        r, c = _parser().split_token_ids(ids)
        assert r == [1, 2]
        assert c == [3, 4]

    def test_start_marker_stripped(self):
        ids = [THINK_START, 1, THINK_END, 3]
        r, c = _parser().split_token_ids(ids)
        assert r == [1]
        assert c == [3]

    def test_no_end_marker_all_reasoning(self):
        # GLM template pre-seeds <think>: output starts mid-reasoning.
        ids = [1, 2, 3]
        r, c = _parser().split_token_ids(ids)
        assert r == [1, 2, 3]
        assert c == []

    def test_reasoning_empty(self):
        ids = [THINK_END, 3, 4]
        r, c = _parser().split_token_ids(ids)
        assert r == []
        assert c == [3, 4]

    def test_invalid_ids_returns_none(self):
        parser = Glm45ReasoningParser(-1, -2)
        assert parser.split_token_ids([1, 2, 3]) is None
        assert not parser.has_token_ids


# ---------------------------------------------------------------------------
# Non-streaming: text fallback
# ---------------------------------------------------------------------------

class TestExtractReasoningText:

    def test_split(self):
        r, c = _parser().extract_reasoning("I think.</think>The answer.")
        assert r == "I think."
        assert c == "The answer."

    def test_leading_start_marker(self):
        r, c = _parser().extract_reasoning(
            "<think>I think.</think>Answer.")
        assert r == "I think."
        assert c == "Answer."

    def test_no_end_marker_all_reasoning(self):
        r, c = _parser().extract_reasoning("only thoughts here")
        assert r == "only thoughts here"
        assert c is None

    def test_reasoning_only(self):
        r, c = _parser().extract_reasoning("thoughts</think>")
        assert r == "thoughts"
        assert c is None

    def test_empty_reasoning(self):
        r, c = _parser().extract_reasoning("</think>Answer.")
        assert r is None
        assert c == "Answer."


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreamingTokenMode:

    def test_boundary_mid_stream(self):
        s = _parser().stream(thinking=True)
        out = []
        out.append(s.process("I am ", [1]))
        out.append(s.process("thinking", [2]))
        out.append(s.process("</think>", [THINK_END]))
        out.append(s.process("The answer", [3]))
        assert out == [
            ("I am ", ""),
            ("thinking", ""),
            ("", ""),
            ("", "The answer"),
        ]
        assert s.finish() == ("", "")

    def test_marker_and_content_in_same_delta(self):
        s = _parser().stream(thinking=True)
        r, c = s.process("done</think>Answer", [2, THINK_END, 3])
        assert r == "done"
        assert c == "Answer"

    def test_start_marker_stripped(self):
        s = _parser().stream(thinking=True)
        r, c = s.process("<think>thought", [THINK_START, 1])
        assert r == "thought"
        assert c == ""

    def test_thinking_disabled_all_content(self):
        s = _parser().stream(thinking=False)
        r, c = s.process("plain text", [1, 2])
        assert r == ""
        assert c == "plain text"

    def test_reasoning_only_stream(self):
        s = _parser().stream(thinking=True)
        r, c = s.process("thoughts forever", [1, 2])
        assert (r, c) == ("thoughts forever", "")
        assert s.finish() == ("", "")


class TestStreamingTextFallback:

    def test_marker_split_across_deltas(self):
        s = Glm45ReasoningParser(-1, -2).stream(thinking=True)
        out1 = s.process("thinking</th", None)
        out2 = s.process("ink>answer", None)
        assert out1 == ("thinking", "")
        assert out2 == ("", "answer")

    def test_partial_marker_flushed_on_finish(self):
        s = Glm45ReasoningParser(-1, -2).stream(thinking=True)
        r, c = s.process("thought</thi", None)
        assert r == "thought"
        assert s.finish() == ("</thi", "")

    def test_leading_start_marker_stripped(self):
        s = Glm45ReasoningParser(-1, -2).stream(thinking=True)
        r, c = s.process("<think>abc</think>d", None)
        assert r == "abc"
        assert c == "d"
