"""Tests for reasoning_mode — parameter overrides inside think blocks."""

import pytest

from orchestrator.context_annotation import ContextAnnotation
from orchestrator.reasoning_mode import (
    ReasoningMode,
    ReasoningModeConfig,
    ReasoningOverrides,
)

START = 128798
END = 128799
OTHER = 42


def _annotation(start: int = START, end: int = END) -> ContextAnnotation:
    return ContextAnnotation(start, end)


def _rm(**kw) -> ReasoningMode:
    return ReasoningMode(config=ReasoningModeConfig(**kw) if kw else None)


def _rm_enabled(**kw) -> ReasoningMode:
    kw.setdefault("enabled", True)
    return ReasoningMode(config=ReasoningModeConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        c = ReasoningModeConfig()
        assert c.enabled is False
        assert c.think_token_detection is True
        assert c.aggressive_speculation_depth_multiplier == 2.0
        assert c.relaxed_verification_threshold == 0.85

    def test_frozen(self):
        c = ReasoningModeConfig()
        with pytest.raises(AttributeError):
            c.enabled = True  # type: ignore[misc]

    def test_custom_values(self):
        c = ReasoningModeConfig(
            enabled=True,
            aggressive_speculation_depth_multiplier=3.0,
            relaxed_verification_threshold=0.80,
        )
        assert c.enabled is True
        assert c.aggressive_speculation_depth_multiplier == 3.0
        assert c.relaxed_verification_threshold == 0.80


# ---------------------------------------------------------------------------
# ReasoningOverrides
# ---------------------------------------------------------------------------

class TestOverridesDataclass:

    def test_inactive_defaults(self):
        o = ReasoningOverrides()
        assert o.active is False
        assert o.depth_multiplier == 1.0
        assert o.adaptive_topk_threshold is None
        assert o.sparse_verification is None

    def test_active_has_values(self):
        o = ReasoningOverrides(
            active=True, depth_multiplier=2.0,
            adaptive_topk_threshold=0.85, sparse_verification=True,
        )
        assert o.active is True
        assert o.depth_multiplier == 2.0
        assert o.adaptive_topk_threshold == 0.85
        assert o.sparse_verification is True

    def test_frozen(self):
        o = ReasoningOverrides()
        with pytest.raises(AttributeError):
            o.active = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_overrides
# ---------------------------------------------------------------------------

class TestComputeOverrides:

    def test_disabled_returns_inactive(self):
        rm = _rm(enabled=False)
        ann = _annotation()
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.active is False
        assert o.depth_multiplier == 1.0

    def test_enabled_not_thinking_returns_inactive(self):
        rm = _rm_enabled()
        ann = _annotation()
        o = rm.compute_overrides(ann)
        assert o.active is False

    def test_enabled_thinking_returns_active(self):
        rm = _rm_enabled()
        ann = _annotation()
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.active is True

    def test_depth_multiplier_from_config(self):
        rm = _rm_enabled(aggressive_speculation_depth_multiplier=3.0)
        ann = _annotation()
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.depth_multiplier == 3.0

    def test_relaxed_threshold_from_config(self):
        rm = _rm_enabled(relaxed_verification_threshold=0.80)
        ann = _annotation()
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.adaptive_topk_threshold == 0.80

    def test_revert_on_think_end(self):
        rm = _rm_enabled()
        ann = _annotation()
        ann.update_token(START)
        assert rm.compute_overrides(ann).active is True
        ann.update_token(END)
        o = rm.compute_overrides(ann)
        assert o.active is False
        assert o.depth_multiplier == 1.0
        assert o.adaptive_topk_threshold is None

    def test_sentinel_token_ids_always_inactive(self):
        rm = _rm_enabled()
        ann = ContextAnnotation(-1, -2)
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.active is False

    def test_sparse_verification_override(self):
        rm = _rm_enabled()
        ann = _annotation()
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.sparse_verification is True


# ---------------------------------------------------------------------------
# apply_depth
# ---------------------------------------------------------------------------

class TestApplyDepth:

    def test_multiplier_applied(self):
        o = ReasoningOverrides(active=True, depth_multiplier=2.0)
        assert ReasoningMode.apply_depth(3, o) == 6

    def test_inactive_no_change(self):
        o = ReasoningOverrides()
        assert ReasoningMode.apply_depth(5, o) == 5

    def test_floor_rounding(self):
        o = ReasoningOverrides(active=True, depth_multiplier=1.5)
        assert ReasoningMode.apply_depth(3, o) == 4

    def test_depth_zero_stays_zero(self):
        o = ReasoningOverrides(active=True, depth_multiplier=2.0)
        assert ReasoningMode.apply_depth(0, o) == 0


# ---------------------------------------------------------------------------
# Integration with ContextAnnotation
# ---------------------------------------------------------------------------

class TestIntegrationWithContextAnnotation:

    def test_full_flow_enter_think(self):
        rm = _rm_enabled()
        ann = _annotation()
        assert rm.compute_overrides(ann).active is False
        ann.update_token(START)
        o = rm.compute_overrides(ann)
        assert o.active is True
        assert o.depth_multiplier == 2.0
        assert o.adaptive_topk_threshold == 0.85

    def test_full_flow_exit_think_revert(self):
        rm = _rm_enabled()
        ann = _annotation()
        ann.update_batch([OTHER, START, OTHER, OTHER])
        assert rm.compute_overrides(ann).active is True
        ann.update_token(END)
        o = rm.compute_overrides(ann)
        assert o.active is False
        assert o.depth_multiplier == 1.0
        assert o.adaptive_topk_threshold is None
        assert o.sparse_verification is None

    def test_multiple_think_blocks(self):
        rm = _rm_enabled()
        ann = _annotation()
        ann.update_token(START)
        assert rm.compute_overrides(ann).active is True
        ann.update_token(END)
        assert rm.compute_overrides(ann).active is False
        ann.update_token(START)
        assert rm.compute_overrides(ann).active is True
        ann.update_token(END)
        assert rm.compute_overrides(ann).active is False

    def test_annotation_with_sentinels_never_active(self):
        rm = _rm_enabled()
        ann = ContextAnnotation()
        for tid in [START, END, START, OTHER]:
            ann.update_token(tid)
            assert rm.compute_overrides(ann).active is False
