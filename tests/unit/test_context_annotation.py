"""Tests for context_annotation — per-request incrementally-maintained metadata."""

from orchestrator.context_annotation import (
    ContextAnnotation,
    ContextAnnotationRegistry,
    ContextAnnotationTracker,
    ReasoningModeTracker,
)

START = 128798
END = 128799
OTHER = 42


# ---------------------------------------------------------------------------
# ReasoningModeTracker
# ---------------------------------------------------------------------------

class TestReasoningModeTracker:

    def test_initial_state(self):
        t = ReasoningModeTracker(START, END)
        assert t.is_thinking is False

    def test_start_token_flips_on(self):
        t = ReasoningModeTracker(START, END)
        t.update_token(START)
        assert t.is_thinking is True

    def test_end_token_flips_off(self):
        t = ReasoningModeTracker(START, END)
        t.update_token(START)
        t.update_token(END)
        assert t.is_thinking is False

    def test_toggle_sequence(self):
        t = ReasoningModeTracker(START, END)
        t.update_token(START)
        assert t.is_thinking is True
        t.update_token(END)
        assert t.is_thinking is False
        t.update_token(START)
        assert t.is_thinking is True

    def test_noop_on_sentinel_ids(self):
        t = ReasoningModeTracker(-1, -2)
        t.update_token(-1)
        assert t.is_thinking is False
        t.update_token(-2)
        assert t.is_thinking is False
        t.update_token(START)
        assert t.is_thinking is False

    def test_rebuild_from_history(self):
        t = ReasoningModeTracker(START, END)
        history = [OTHER, START, OTHER, OTHER, END, OTHER, START, OTHER]
        t.rebuild(history)
        assert t.is_thinking is True

    def test_rebuild_empty(self):
        t = ReasoningModeTracker(START, END)
        t.update_token(START)
        t.rebuild([])
        assert t.is_thinking is False

    def test_unrelated_tokens_ignored(self):
        t = ReasoningModeTracker(START, END)
        for _ in range(100):
            t.update_token(OTHER)
        assert t.is_thinking is False

    def test_reset(self):
        t = ReasoningModeTracker(START, END)
        t.update_token(START)
        assert t.is_thinking is True
        t.reset()
        assert t.is_thinking is False


# ---------------------------------------------------------------------------
# ContextAnnotation (composite)
# ---------------------------------------------------------------------------

class TestContextAnnotation:

    def test_reasoning_member_accessible(self):
        ca = ContextAnnotation(START, END)
        assert isinstance(ca.reasoning, ReasoningModeTracker)
        assert ca.reasoning.is_thinking is False

    def test_delegates_update_token(self):
        ca = ContextAnnotation(START, END)
        ca.update_token(START)
        assert ca.reasoning.is_thinking is True

    def test_delegates_update_batch(self):
        ca = ContextAnnotation(START, END)
        ca.update_batch([OTHER, START, OTHER])
        assert ca.reasoning.is_thinking is True

    def test_delegates_rebuild(self):
        ca = ContextAnnotation(START, END)
        ca.update_token(START)
        ca.rebuild([OTHER, START, END])
        assert ca.reasoning.is_thinking is False

    def test_delegates_reset(self):
        ca = ContextAnnotation(START, END)
        ca.update_token(START)
        ca.reset()
        assert ca.reasoning.is_thinking is False

    def test_factory_with_sentinels(self):
        ca = ContextAnnotation()
        ca.update_token(START)
        assert ca.reasoning.is_thinking is False


# ---------------------------------------------------------------------------
# ContextAnnotationRegistry
# ---------------------------------------------------------------------------

class TestContextAnnotationRegistry:

    def test_starts_empty(self):
        reg = ContextAnnotationRegistry(START, END)
        assert len(reg) == 0

    def test_get_or_create_new(self):
        reg = ContextAnnotationRegistry(START, END)
        ca = reg.get_or_create(1)
        assert isinstance(ca, ContextAnnotation)
        assert len(reg) == 1

    def test_get_or_create_existing_returns_same(self):
        reg = ContextAnnotationRegistry(START, END)
        ca1 = reg.get_or_create(1)
        ca2 = reg.get_or_create(1)
        assert ca1 is ca2

    def test_remove_drops_entry(self):
        reg = ContextAnnotationRegistry(START, END)
        reg.get_or_create(1)
        reg.remove(1)
        assert len(reg) == 0
        assert 1 not in reg

    def test_remove_nonexistent_is_safe(self):
        reg = ContextAnnotationRegistry(START, END)
        reg.remove(999)
        assert len(reg) == 0

    def test_multiple_requests_independent(self):
        reg = ContextAnnotationRegistry(START, END)
        ca1 = reg.get_or_create(1)
        ca2 = reg.get_or_create(2)
        ca1.update_token(START)
        assert ca1.reasoning.is_thinking is True
        assert ca2.reasoning.is_thinking is False

    def test_get_returns_none_for_missing(self):
        reg = ContextAnnotationRegistry(START, END)
        assert reg.get(42) is None

    def test_get_returns_existing(self):
        reg = ContextAnnotationRegistry(START, END)
        ca = reg.get_or_create(1)
        assert reg.get(1) is ca

    def test_contains(self):
        reg = ContextAnnotationRegistry(START, END)
        assert 1 not in reg
        reg.get_or_create(1)
        assert 1 in reg


# ---------------------------------------------------------------------------
# Batch equivalence
# ---------------------------------------------------------------------------

class TestUpdateBatch:

    def test_batch_equivalent_to_sequential(self):
        t1 = ReasoningModeTracker(START, END)
        t2 = ReasoningModeTracker(START, END)
        tokens = [OTHER, START, OTHER, END, START]
        for tid in tokens:
            t1.update_token(tid)
        t2.update_batch(tokens)
        assert t1.is_thinking == t2.is_thinking

    def test_empty_batch_noop(self):
        t = ReasoningModeTracker(START, END)
        t.update_batch([])
        assert t.is_thinking is False

    def test_mixed_tokens(self):
        t = ReasoningModeTracker(START, END)
        t.update_batch([START, OTHER, END, OTHER, OTHER])
        assert t.is_thinking is False


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocol:

    def test_tracker_satisfies_protocol(self):
        t = ReasoningModeTracker(START, END)
        assert isinstance(t, ContextAnnotationTracker)

    def test_context_annotation_satisfies_protocol(self):
        ca = ContextAnnotation(START, END)
        assert isinstance(ca, ContextAnnotationTracker)
