"""Guided-decoding unit tests (TD-SERVE-NAMED-TOOL-CHOICE) — REAL xgrammar
against the REAL GLM-5.2 tokenizer, no engine.

Covers: grammar compilation from tool schemas (named / "required" union /
typed args), token-mask correctness on scripted logits (forced tokens
only), the think-prefix interplay (reasoning=True embeds the free-form
``<think>``-section in the grammar: EOS suppressed, wire tags excluded,
grammar engages at ``</think>``), grammar completion → the orchestrator's
"tool_calls" stop, and the rollback protocol used by the speculative
draft-truncation predicate (STAGE B).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

xgr = pytest.importorskip("xgrammar")
transformers = pytest.importorskip("transformers")

_TOKENIZER_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
                  / "test-data" / "GLM-5.2")
pytestmark = pytest.mark.skipif(
    not (_TOKENIZER_DIR / "tokenizer.json").exists(),
    reason="GLM-5.2 tokenizer files not present")

VOCAB = 154880          # GLM-5.2 model vocab (config.json vocab_size)

_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}
_MAIL = {
    "type": "function",
    "function": {
        "name": "send_mail",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}},
            "required": ["to"],
        },
    },
}
_NAMED = {"type": "function", "function": {"name": "get_weather"}}


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(
        str(_TOKENIZER_DIR), trust_remote_code=True)


@pytest.fixture(scope="module")
def manager(tok):
    from server.guided import GuidedDecodingManager
    return GuidedDecodingManager(tok, VOCAB)


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def _accept_all(state, ids) -> bool:
    return all(state.try_accept(t) for t in ids)


_CALL = ("<tool_call>get_weather<arg_key>city</arg_key>"
         "<arg_value>Paris</arg_value><arg_key>days</arg_key>"
         "<arg_value>3</arg_value></tool_call>")


class TestGrammarCompilation:

    def test_named_accepts_canonical_call(self, manager, tok):
        st = manager.build([_WEATHER, _MAIL], _NAMED, reasoning=False)
        s = st.new_state()
        assert _accept_all(s, _ids(tok, _CALL))
        assert s.completed

    def test_named_rejects_other_function(self, manager, tok):
        s = manager.build([_WEATHER, _MAIL], _NAMED,
                          reasoning=False).new_state()
        ids = _ids(tok, "<tool_call>send_mail")
        assert s.try_accept(ids[0])          # <tool_call> tag token
        assert not s.try_accept(ids[1])      # wrong function name

    def test_required_union_accepts_both(self, manager, tok):
        g = manager.build([_WEATHER, _MAIL], "required", reasoning=False)
        for call in (_CALL,
                     "<tool_call>send_mail<arg_key>to</arg_key>"
                     "<arg_value>bob</arg_value></tool_call>"):
            s = g.new_state()
            assert _accept_all(s, _ids(tok, call)), call
            assert s.completed

    def test_typed_args_reject_non_integer(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        prefix = ("<tool_call>get_weather<arg_key>city</arg_key>"
                  "<arg_value>Paris</arg_value><arg_key>days</arg_key>"
                  "<arg_value>")
        assert _accept_all(s, _ids(tok, prefix))
        bad = _ids(tok, "soon")
        assert not _accept_all(s, bad), \
            "integer-typed arg_value must reject non-numeric text"

    def test_grammar_cache_reuse(self, manager):
        g1 = manager.build([_WEATHER], _NAMED, reasoning=False)
        g2 = manager.build([_WEATHER], _NAMED, reasoning=False)
        assert g1 is g2
        g3 = manager.build([_WEATHER], _NAMED, reasoning=True)
        assert g3 is not g1


class TestMaskCorrectness:

    def test_forced_tokens_only_at_start(self, manager, tok):
        """reasoning=False named grammar: position 0 allows exactly the
        tokenizations of the "<tool_call>" opener — a scripted logits
        vector peaking at an arbitrary banned token must be repaired to
        the highest-scoring ALLOWED token."""
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        tool_call_tok = _ids(tok, "<tool_call>")[0]
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[12345] = 10.0                  # invalid peak
        logits[tool_call_tok] = 1.0
        pick = s.pick_and_accept(logits)
        assert pick == tool_call_tok

    def test_greedy_pick_follows_allowed_argmax(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        ids = _ids(tok, _CALL)
        for i, t in enumerate(ids):
            logits = np.zeros(VOCAB, dtype=np.float32)
            logits[t] = 5.0
            pick = s.pick_and_accept(logits)
            assert pick == t, f"position {i}: pick {pick} != scripted {t}"
        assert s.completed

    def test_completion_mask_is_stop_only(self, manager, tok):
        """After the forced call closes, only stop tokens remain — the
        orchestrator stops BEFORE sampling there (finish tool_calls), so
        no trailing junk can be emitted."""
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        assert _accept_all(s, _ids(tok, _CALL))
        allowed = np.flatnonzero(s._allowed_bits())
        assert len(allowed) <= 2 and 154820 in allowed  # <|endoftext|>

    def test_sampled_pick_respects_mask(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        tool_call_tok = _ids(tok, "<tool_call>")[0]
        logits = np.random.default_rng(0).normal(
            size=VOCAB).astype(np.float32) * 3.0
        pick = s.pick_and_accept(logits, temperature=0.8, top_p=0.9,
                                 top_k=40, seed=7)
        allowed_before = {27, 62311, tool_call_tok}  # '<', '<t', <tool_call>
        assert pick in allowed_before


class TestThinkInterplay:

    def test_free_form_think_then_forced_call(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=True).new_state()
        think = _ids(tok, "Let me check the weather for the user.")
        assert _accept_all(s, think)          # free-form inside <think>
        assert _accept_all(s, _ids(tok, "</think>"))
        assert _accept_all(s, _ids(tok, _CALL))
        assert s.completed

    def test_eos_suppressed_inside_think(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=True).new_state()
        assert _accept_all(s, _ids(tok, "thinking..."))
        assert not s.try_accept(154820), \
            "EOS must be masked until the forced call completes"

    def test_wire_tags_excluded_inside_think(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=True).new_state()
        tool_call_tok = _ids(tok, "<tool_call>")[0]
        assert not s.try_accept(tool_call_tok), \
            "<tool_call> inside the think section must be a violation"

    def test_grammar_engages_at_think_end(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=True).new_state()
        assert _accept_all(s, _ids(tok, "ok</think>"))
        # After </think> only the forced-call opener is allowed: a plain
        # newline (the model's natural next token) is a violation the
        # masked step repairs.
        nl = _ids(tok, "\n")
        assert len(nl) == 1 and not s.try_accept(nl[0])
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[nl[0]] = 10.0
        pick = s.pick_and_accept(logits)
        assert pick in (27, 62311, _ids(tok, "<tool_call>")[0])


class TestRollbackProtocol:

    def test_tentative_advance_rollback_restores_state(self, manager, tok):
        """The STAGE B draft-truncation predicate: tentatively accept a
        valid prefix, rollback, then the SAME tokens must accept again
        (matcher state restored exactly)."""
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        ids = _ids(tok, _CALL)
        head, tail = ids[:4], ids[4:]
        assert _accept_all(s, head)
        # Tentative advance over 5 tokens of the tail, then rollback.
        n_ok = 0
        for t in tail[:5]:
            if not s.try_accept(t):
                break
            n_ok += 1
        assert n_ok == 5
        s.rollback(n_ok)
        # Real advance must replay identically to completion.
        assert _accept_all(s, tail)
        assert s.completed

    def test_rejection_leaves_matcher_usable(self, manager, tok):
        s = manager.build([_WEATHER], _NAMED, reasoning=False).new_state()
        ids = _ids(tok, _CALL)
        assert s.try_accept(ids[0])
        assert not s.try_accept(154820)        # violation: EOS mid-call
        assert _accept_all(s, ids[1:])         # continues unharmed
        assert s.completed
