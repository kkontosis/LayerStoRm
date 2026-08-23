"""Tests for orchestrator.prompt_lookup — N-gram suffix matching speculation."""

import pytest

from orchestrator.prompt_lookup import PromptLookup, PromptLookupConfig


# ---------------------------------------------------------------------------
# N-gram matching
# ---------------------------------------------------------------------------


class TestNgramMatch:

    def test_bigram_match_returns_continuation(self):
        pl = PromptLookup(PromptLookupConfig(min_ngram_size=2, max_ngram_size=2))
        tokens = [1, 2, 3, 4, 5, 1, 2]
        result = pl.lookup(tokens)
        # Match [1,2] at position 0, continuation is [3,4,5,1,2] capped to 5
        assert result == [3, 4, 5, 1, 2]

    def test_trigram_match_returns_continuation(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=3, max_ngram_size=3, max_continuation_length=2))
        tokens = [10, 20, 30, 40, 50, 10, 20, 30]
        result = pl.lookup(tokens)
        assert result == [40, 50]

    def test_no_match_returns_empty(self):
        pl = PromptLookup(PromptLookupConfig(min_ngram_size=2, max_ngram_size=4))
        tokens = [1, 2, 3, 4, 5]
        result = pl.lookup(tokens)
        assert result == []

    def test_longest_match_wins(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=4, max_continuation_length=10))
        # Suffix of length 3 matches at position 0: [1,2,3] -> continuation [99,1,2,3]
        # Suffix of length 2 would match at position 5: [2,3] -> continuation [1,2,3]
        # Greedy longest should pick the 3-gram.
        tokens = [1, 2, 3, 99, 1, 2, 3]
        result = pl.lookup(tokens)
        assert result == [99, 1, 2, 3]

    def test_most_recent_match_preferred(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=10))
        # Two bigram matches for suffix [1,2]: at position 0 and position 4.
        # Most recent (position 4) should win -> continuation [300, 1, 2]
        tokens = [1, 2, 100, 200, 1, 2, 300, 1, 2]
        result = pl.lookup(tokens)
        assert result == [300, 1, 2]

    def test_max_continuation_length_caps_output(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=2))
        tokens = [1, 2, 3, 4, 5, 6, 1, 2]
        result = pl.lookup(tokens)
        assert result == [3, 4]
        assert len(result) <= 2

    def test_max_continuation_parameter_overrides_config(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=10))
        tokens = [1, 2, 3, 4, 5, 6, 1, 2]
        result = pl.lookup(tokens, max_continuation=1)
        assert result == [3]


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------


class TestDisabled:

    def test_disabled_returns_empty(self):
        pl = PromptLookup(PromptLookupConfig(enabled=False))
        tokens = [1, 2, 3, 1, 2]
        result = pl.lookup(tokens)
        assert result == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_sequence(self):
        pl = PromptLookup()
        assert pl.lookup([]) == []

    def test_single_token(self):
        pl = PromptLookup()
        assert pl.lookup([42]) == []

    def test_sequence_too_short_for_min_ngram(self):
        pl = PromptLookup(PromptLookupConfig(min_ngram_size=3))
        # Need at least min_ngram_size + 1 = 4 tokens
        assert pl.lookup([1, 2, 3]) == []

    def test_fixed_ngram_size(self):
        pl = PromptLookup(PromptLookupConfig(min_ngram_size=3, max_ngram_size=3))
        tokens = [1, 2, 3, 99, 1, 2, 3]
        result = pl.lookup(tokens)
        assert result == [99, 1, 2, 3]

    def test_all_same_tokens(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=3))
        tokens = [7, 7, 7, 7, 7, 7]
        result = pl.lookup(tokens)
        assert len(result) > 0
        assert all(t == 7 for t in result)

    def test_continuation_truncated_at_end_of_sequence(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=100))
        # Match [5,6] at position 0, continuation extends to end: [9,5,6]
        tokens = [5, 6, 9, 5, 6]
        result = pl.lookup(tokens)
        assert result == [9, 5, 6]

    def test_match_with_overlap(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2, max_continuation_length=10))
        # Suffix [3,4] matches at position 1: [3,4] -> continuation [3,4]
        tokens = [1, 3, 4, 3, 4]
        result = pl.lookup(tokens)
        assert result == [3, 4]


# ---------------------------------------------------------------------------
# Acceptance rate tracking
# ---------------------------------------------------------------------------


class TestAcceptanceRate:

    def test_initial_acceptance_rate_zero(self):
        pl = PromptLookup()
        assert pl.acceptance_rate == 0.0

    def test_first_record_sets_rate(self):
        pl = PromptLookup()
        pl.record_result(10, 8)
        assert pl.acceptance_rate == pytest.approx(0.8)

    def test_ema_smoothing(self):
        pl = PromptLookup(PromptLookupConfig(acceptance_ema_alpha=0.5))
        pl.record_result(10, 10)  # rate = 1.0
        assert pl.acceptance_rate == pytest.approx(1.0)
        pl.record_result(10, 0)   # rate = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert pl.acceptance_rate == pytest.approx(0.5)
        pl.record_result(10, 0)   # rate = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        assert pl.acceptance_rate == pytest.approx(0.25)

    def test_zero_proposed_no_update(self):
        pl = PromptLookup()
        pl.record_result(10, 8)
        rate_before = pl.acceptance_rate
        pl.record_result(0, 0)
        assert pl.acceptance_rate == rate_before


# ---------------------------------------------------------------------------
# Match rate tracking
# ---------------------------------------------------------------------------


class TestMatchRate:

    def test_initial_match_rate_zero(self):
        pl = PromptLookup()
        assert pl.match_rate == 0.0

    def test_match_rate_after_lookups(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2))
        tokens_match = [1, 2, 3, 1, 2]
        tokens_no_match = [1, 2, 3, 4, 5]

        pl.lookup(tokens_match)      # match
        pl.lookup(tokens_no_match)   # no match
        pl.lookup(tokens_match)      # match
        pl.lookup(tokens_no_match)   # no match
        pl.lookup(tokens_match)      # match

        assert pl.match_rate == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


class TestCounters:

    def test_total_lookups_incremented(self):
        pl = PromptLookup()
        pl.lookup([1, 2, 3])
        pl.lookup([4, 5, 6])
        pl.lookup([7, 8, 9])
        assert pl.total_lookups == 3

    def test_total_matches_only_on_match(self):
        pl = PromptLookup(PromptLookupConfig(
            min_ngram_size=2, max_ngram_size=2))
        pl.lookup([1, 2, 3, 1, 2])   # match
        pl.lookup([1, 2, 3, 4, 5])   # no match
        assert pl.total_matches == 1

    def test_proposed_and_accepted_accumulate(self):
        pl = PromptLookup()
        pl.record_result(5, 3)
        pl.record_result(10, 7)
        assert pl.total_proposed == 15
        assert pl.total_accepted == 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = PromptLookupConfig()
        assert cfg.enabled is True
        assert cfg.min_ngram_size == 2
        assert cfg.max_ngram_size == 4
        assert cfg.max_continuation_length == 5
        assert cfg.acceptance_ema_alpha == pytest.approx(0.3)

    def test_frozen_config(self):
        cfg = PromptLookupConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_is_enabled_property(self):
        assert PromptLookup(PromptLookupConfig(enabled=True)).is_enabled is True
        assert PromptLookup(PromptLookupConfig(enabled=False)).is_enabled is False

    def test_disabled_still_increments_lookups(self):
        pl = PromptLookup(PromptLookupConfig(enabled=False))
        pl.lookup([1, 2, 3, 1, 2])
        assert pl.total_lookups == 1
        assert pl.total_matches == 0
