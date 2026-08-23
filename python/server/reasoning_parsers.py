"""Reasoning ("thinking") parsers — split reasoning content from final content.

Registry pattern mirroring vLLM's ReasoningParserManager, kept lean: parsers
are registered by CLI/config name (``serving.reasoning_parser`` /
``--reasoning-parser``) and split model output into ``reasoning_content``
vs ``content`` for both non-streaming and streaming (SSE delta) responses.

Boundary detection is TOKEN-ID based where the tokenizer detected dedicated
think-marker token ids (GLM-5.2: ``<think>``=154841, ``</think>``=154842);
the text-marker path is only a fallback for tokenizers whose markers are not
single tokens.

The ``glm45`` parser semantics are ported from ref/vllm (Apache-2.0,
Copyright contributors to the vLLM project — see THIRD_PARTY_NOTICES.md):
  - ref/vllm/vllm/reasoning/basic_parsers.py (BaseThinkingReasoningParser:
    reasoning-first assumption when the start marker is absent — GLM chat
    templates pre-seed ``<think>`` in the generation prompt, so model output
    begins mid-reasoning)
  - ref/vllm/vllm/parser/glm47_moe.py (thinking_enabled gating: with
    thinking disabled the template emits ``<think></think>`` in the PROMPT,
    so the output carries no markers and is all content)
vLLM registers ``glm45`` and ``glm47`` to the same GLM parser; we mirror
that aliasing.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "DeepSeekV4ReasoningParser",
    "Glm45ReasoningParser",
    "ReasoningStream",
    "get_reasoning_parser",
    "reasoning_parser_names",
]


def _partial_marker_len(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of
    ``marker`` (used to withhold a possibly-incomplete marker in the
    text-fallback streaming path)."""
    max_k = min(len(text), len(marker) - 1)
    for k in range(max_k, 0, -1):
        if marker.startswith(text[-k:]):
            return k
    return 0


class ReasoningStream:
    """Stateful per-request streaming splitter.

    ``process(delta_text, delta_token_ids)`` routes each decoded delta into
    ``(reasoning_delta, content_delta)``.  Token-id boundary detection is
    used when the parser has valid marker ids AND ids are supplied;
    otherwise falls back to buffered text-marker scanning.
    """

    def __init__(
        self,
        *,
        start_str: str,
        end_str: str,
        start_id: int,
        end_id: int,
        thinking: bool = True,
    ) -> None:
        self._start_str = start_str
        self._end_str = end_str
        self._start_id = start_id
        self._end_id = end_id
        self._in_reasoning = thinking
        self._buf = ""  # text-fallback withhold buffer (reasoning side)

    @property
    def in_reasoning(self) -> bool:
        return self._in_reasoning

    def process(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int] | None = None,
    ) -> tuple[str, str]:
        """Route a decoded delta. Returns ``(reasoning_delta, content_delta)``."""
        if not self._in_reasoning:
            return "", delta_text
        if not delta_text and not delta_token_ids:
            return "", ""

        token_mode = (
            self._end_id >= 0 and delta_token_ids is not None
        )
        if token_mode:
            # Skip/strip a start marker the model chose to emit anyway.
            if self._start_id >= 0 and self._start_id in delta_token_ids:
                delta_text = delta_text.replace(self._start_str, "", 1)
            if self._end_id in delta_token_ids:
                reasoning, sep, content = delta_text.partition(self._end_str)
                self._in_reasoning = False
                if not sep:
                    # Marker token present but its text is not (e.g. stop
                    # truncation) — everything remaining is reasoning.
                    return reasoning, ""
                return reasoning, content
            return delta_text, ""

        # Text fallback: buffer so a marker split across deltas is caught.
        self._buf += delta_text
        idx = self._buf.find(self._end_str)
        if idx >= 0:
            reasoning = self._buf[:idx]
            content = self._buf[idx + len(self._end_str):]
            self._buf = ""
            self._in_reasoning = False
            if reasoning.startswith(self._start_str):
                reasoning = reasoning[len(self._start_str):]
            return reasoning, content
        hold = _partial_marker_len(self._buf, self._end_str)
        emit = self._buf[: len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:]
        if emit.startswith(self._start_str):
            emit = emit[len(self._start_str):]
        return emit, ""

    def finish(self) -> tuple[str, str]:
        """Flush withheld text at end of stream."""
        emit, self._buf = self._buf, ""
        if self._in_reasoning:
            return emit, ""
        return "", emit


class Glm45ReasoningParser:
    """GLM-4.5/4.7/5.x ``<think>...</think>`` reasoning parser.

    Ported (lean) from ref/vllm/vllm/reasoning/basic_parsers.py and
    ref/vllm/vllm/parser/glm47_moe.py — see module docstring.
    """

    name = "glm45"
    start_str = "<think>"
    end_str = "</think>"

    def __init__(self, think_start_token_id: int = -1,
                 think_end_token_id: int = -1, *,
                 reasoning_start_str: str | None = None,
                 reasoning_end_str: str | None = None) -> None:
        """``reasoning_start_str`` / ``reasoning_end_str`` (the vLLM
        ReasoningConfig field names, carried by --reasoning-config) override
        the parser's class markers per instance.  Overridden markers force
        the TEXT path — the tokenizer-detected think token ids match the
        default markers only, so they are dropped (fail-safe: the text
        scanner honors exactly the strings the user configured)."""
        self._start_id = think_start_token_id
        self._end_id = think_end_token_id
        if reasoning_start_str is not None:
            self.start_str = reasoning_start_str        # instance shadow
        if reasoning_end_str is not None:
            self.end_str = reasoning_end_str
        if reasoning_start_str is not None or reasoning_end_str is not None:
            self._start_id = -1
            self._end_id = -1

    @property
    def has_token_ids(self) -> bool:
        return self._end_id >= 0

    def split_token_ids(
        self, token_ids: Sequence[int],
    ) -> tuple[list[int], list[int]] | None:
        """Token-id split: ``(reasoning_ids, content_ids)`` with marker
        tokens removed.  ``None`` when marker ids are unknown (caller should
        use the text fallback).  No end marker → all reasoning (the GLM
        template pre-seeds ``<think>``, so generation starts mid-reasoning).
        """
        if self._end_id < 0:
            return None
        ids = list(token_ids)
        if self._end_id in ids:
            cut = ids.index(self._end_id)
            reasoning_ids = ids[:cut]
            content_ids = ids[cut + 1:]
        else:
            reasoning_ids = ids
            content_ids = []
        if self._start_id >= 0:
            reasoning_ids = [t for t in reasoning_ids if t != self._start_id]
        return reasoning_ids, content_ids

    def extract_reasoning(
        self, text: str,
    ) -> tuple[str | None, str | None]:
        """Text-fallback split. Mirrors vLLM BaseThinkingReasoningParser.
        extract_reasoning: strip a leading start marker; no end marker →
        the whole output is reasoning.  An empty start marker (structural
        boundary — deepseek_v4) has nothing to strip."""
        if self.start_str:
            before, sep, after = text.partition(self.start_str)
            text = after if sep else before
        if self.end_str not in text:
            return (text or None), None
        reasoning, _, content = text.partition(self.end_str)
        return (reasoning or None), (content or None)

    def stream(self, *, thinking: bool = True) -> ReasoningStream:
        return ReasoningStream(
            start_str=self.start_str,
            end_str=self.end_str,
            start_id=self._start_id,
            end_id=self._end_id,
            thinking=thinking,
        )


class DeepSeekV4ReasoningParser(Glm45ReasoningParser):
    """DeepSeek-V4 reasoning parser — STRUCTURAL start boundary.

    Ported (lean) from ref/vllm (Apache-2.0, Copyright contributors to
    the vLLM project): vllm/parser/deepseek_v4.py (initial parser state =
    REASONING when thinking is enabled) via
    vllm/reasoning/deepseek_v4_engine_reasoning_parser.py.

    The V4 chat template pre-opens the think section in the PROMPT
    (generation prompt ends with ``<think>`` when thinking, ``</think>``
    when not), so model output BEGINS inside reasoning and never emits an
    opening marker: the start marker is the EMPTY string (structural
    boundary) and only ``</think>`` closes the section.  ``<think>`` is
    still absorbed if the model emits a spurious one (token-id path, via
    the tokenizer-detected start id — vLLM's (REASONING, THINK_START)
    absorb transition).
    """

    name = "deepseek_v4"
    start_str = ""            # structural: output begins in reasoning
    end_str = "</think>"


_REASONING_PARSERS: dict[str, type[Glm45ReasoningParser]] = {
    "glm45": Glm45ReasoningParser,
    "glm47": Glm45ReasoningParser,  # vLLM aliases both to the GLM parser
    "deepseek_v4": DeepSeekV4ReasoningParser,
}


def reasoning_parser_names() -> list[str]:
    return sorted(_REASONING_PARSERS)


def get_reasoning_parser(name: str) -> type[Glm45ReasoningParser]:
    try:
        return _REASONING_PARSERS[name]
    except KeyError:
        raise ValueError(
            f"unknown reasoning parser '{name}' "
            f"(available: {', '.join(reasoning_parser_names())})",
        ) from None
