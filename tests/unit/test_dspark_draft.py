"""Unit tests for the DSpark draft planner (DSP-5) + DraftCombiner source.

The draft forward itself is engine machinery (D_CMD_RUN_DSPARK_STEP,
validated by the DsparkLossless golden); here the PLANNER policy is tested
in isolation: γ clamps, the anchor-position convention, the DSP-6
confidence proxy, and acceptance/EMA tracking — plus the "dspark" source
registration in DraftCombiner.
"""

from __future__ import annotations

from orchestrator.draft_combiner import DraftCombiner, DraftCombinerConfig
from orchestrator.dspark_draft import (
    DsparkDraft,
    DsparkDraftConfig,
    DsparkDraftResult,
)
from orchestrator.mtp_draft import DraftStep, MtpDraftResult


# ---------------------------------------------------------------------------
# DsparkDraft planner
# ---------------------------------------------------------------------------

class TestDsparkDraftPolicy:
    def test_disabled_by_default(self):
        # enabled derives from speculation.method == dspark — a
        # default-constructed planner must be inert.
        d = DsparkDraft()
        assert not d.is_enabled

    def test_gamma_clamped_to_block_size(self):
        d = DsparkDraft(DsparkDraftConfig(
            enabled=True, block_size=8, speculative_tokens=7))
        assert d.gamma == 7
        # speculative_tokens beyond the checkpoint block is clamped
        d2 = DsparkDraft(DsparkDraftConfig(
            enabled=True, block_size=4, speculative_tokens=7))
        assert d2.gamma == 4
        # degenerate config still proposes at least one token
        d3 = DsparkDraft(DsparkDraftConfig(
            enabled=True, block_size=8, speculative_tokens=0))
        assert d3.gamma == 1

    def test_plan_num_query_clamped_by_depth(self):
        d = DsparkDraft(DsparkDraftConfig(enabled=True))
        assert d.plan_num_query(2) == 2   # utility depth below γ
        assert d.plan_num_query(100) == d.gamma
        assert d.plan_num_query(0) == 0
        assert d.plan_num_query(-1) == 0

    def test_anchor_pos_is_fed_token_count(self):
        # Matches the AR dispatch convention (#91 fixed position math):
        # the newest committed token is always still pending, so the
        # fed-token count — which is also the runtime's ingested context
        # length (one row per fed position) — is prompt_len +
        # tokens_generated - 1, clamped at 0.
        d = DsparkDraft(DsparkDraftConfig(enabled=True))
        assert d.anchor_pos(prompt_len=5, tokens_generated=1) == 5
        assert d.anchor_pos(prompt_len=3, tokens_generated=2) == 4
        assert d.anchor_pos(prompt_len=0, tokens_generated=0) == 0

    def test_confidence_proxy_prior_then_ema(self):
        d = DsparkDraft(DsparkDraftConfig(
            enabled=True, acceptance_ema_alpha=0.5))
        # Uninformed prior before any verified round (DSP-6 replaces this
        # proxy with the trained per-position survival head).
        assert d.confidence_proxy() == 0.5
        d.record_result(DsparkDraftResult(tokens=[1, 2, 3, 4]), 4)
        assert d.confidence_proxy() == 1.0
        d.record_result(DsparkDraftResult(tokens=[1, 2, 3, 4]), 0)
        assert abs(d.confidence_proxy() - 0.5) < 1e-9  # EMA halfway


class TestDsparkDraftAcceptanceTracking:
    def test_record_result_totals_and_ema(self):
        d = DsparkDraft(DsparkDraftConfig(
            enabled=True, acceptance_ema_alpha=0.3))
        assert d.acceptance_rate == 0.0
        d.record_result(DsparkDraftResult(tokens=[7, 8]), 2)
        assert d.total_rounds == 1
        assert d.total_proposed == 2
        assert d.total_accepted == 2
        assert d.acceptance_rate == 1.0
        d.record_result(DsparkDraftResult(tokens=[7, 8, 9, 10]), 1)
        assert d.total_rounds == 2
        assert d.total_proposed == 6
        assert d.total_accepted == 3
        # EMA: 0.3 * 0.25 + 0.7 * 1.0
        assert abs(d.acceptance_rate - 0.775) < 1e-9

    def test_accepted_clamped_to_depth(self):
        d = DsparkDraft(DsparkDraftConfig(enabled=True))
        d.record_result(DsparkDraftResult(tokens=[1]), 5)  # over-report
        assert d.total_accepted == 1
        d.record_result(DsparkDraftResult(tokens=[1]), -2)  # under-report
        assert d.total_accepted == 1

    def test_empty_round_does_not_touch_ema(self):
        d = DsparkDraft(DsparkDraftConfig(enabled=True))
        d.record_result(DsparkDraftResult(tokens=[]), 0)
        assert d.total_rounds == 1
        assert d.acceptance_rate == 0.0
        assert d.confidence_proxy() == 0.5  # still the prior


# ---------------------------------------------------------------------------
# DraftCombiner "dspark" source
# ---------------------------------------------------------------------------

class TestDraftCombinerDsparkSource:
    def test_dspark_tokens_selected(self):
        c = DraftCombiner()
        combined = c.combine(
            prompt_lookup_tokens=[],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=4,
            max_verifiable_depth=4,
            dspark_result=DsparkDraftResult(tokens=[100, 101, 102]),
            dspark_confidence=0.5,
        )
        assert combined.tokens == [100, 101, 102]
        assert all(s.source == "dspark" for s in combined.steps)
        assert combined.source_breakdown == {"dspark": 3}
        assert c.source_selection_rates["dspark"] == 1.0

    def test_dspark_depth_clamped_by_ceiling(self):
        c = DraftCombiner()
        combined = c.combine(
            prompt_lookup_tokens=[],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=2,
            max_verifiable_depth=4,
            dspark_result=DsparkDraftResult(tokens=[100, 101, 102]),
            dspark_confidence=0.5,
        )
        assert combined.tokens == [100, 101]

    def test_higher_confidence_source_wins_per_position(self):
        # MTP with high confidence at position 0 beats the low DSpark
        # proxy; DSpark still covers positions MTP did not draft.
        c = DraftCombiner(DraftCombinerConfig(
            mtp_compute_cost=0.05, dspark_compute_cost=0.05))
        mtp = MtpDraftResult(steps=[
            DraftStep(token_id=55, confidence=0.95, mtp_layer_idx=61),
        ])
        combined = c.combine(
            prompt_lookup_tokens=[],
            mtp_result=mtp,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=3,
            max_verifiable_depth=3,
            dspark_result=DsparkDraftResult(tokens=[100, 101, 102]),
            dspark_confidence=0.4,
        )
        assert combined.tokens == [55, 101, 102]
        assert combined.steps[0].source == "mtp"
        assert combined.steps[1].source == "dspark"

    def test_no_dspark_result_unchanged(self):
        # dspark_result defaults to None — pre-DSP-5 call sites and
        # behavior stay identical.
        c = DraftCombiner()
        combined = c.combine(
            prompt_lookup_tokens=[42],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.5,
            max_depth=2,
            max_verifiable_depth=2,
        )
        assert combined.tokens == [42]
        assert combined.steps[0].source == "prompt_lookup"


# ---------------------------------------------------------------------------
# DSP-6: trained confidence head — cumulative survival (INV-DSPARK-CONF)
# ---------------------------------------------------------------------------

class TestDsparkSurvivalConfidences:
    def _draft(self, **kw):
        cfg = dict(enabled=True, block_size=8, speculative_tokens=7,
                   confidence_enabled=True)
        cfg.update(kw)
        return DsparkDraft(DsparkDraftConfig(**cfg))

    def test_cumprod_monotone_non_increasing(self):
        d = self._draft()
        res = DsparkDraftResult(tokens=[1, 2, 3, 4],
                                confidences=[0.9, 0.8, 0.5, 0.99])
        a = d.survival_confidences(res)
        assert a is not None and len(a) == 4
        # a_j = prod_{i<=j} c_i, each in (0, 1], non-increasing
        assert abs(a[0] - 0.9) < 1e-12
        assert abs(a[1] - 0.9 * 0.8) < 1e-12
        assert abs(a[2] - 0.9 * 0.8 * 0.5) < 1e-12
        assert abs(a[3] - 0.9 * 0.8 * 0.5 * 0.99) < 1e-12
        assert all(a[j] <= a[j - 1] for j in range(1, 4))
        assert all(0.0 < x <= 1.0 for x in a)

    def test_none_when_head_disabled(self):
        d = self._draft(confidence_enabled=False)
        res = DsparkDraftResult(tokens=[1, 2], confidences=[0.9, 0.8])
        assert d.survival_confidences(res) is None

    def test_none_when_confidences_missing_or_short(self):
        d = self._draft()
        assert d.survival_confidences(None) is None
        assert d.survival_confidences(
            DsparkDraftResult(tokens=[1, 2])) is None
        # fewer confidences than tokens -> unusable, fall back to proxy
        assert d.survival_confidences(
            DsparkDraftResult(tokens=[1, 2, 3],
                              confidences=[0.9, 0.8])) is None

    def test_out_of_range_ck_clamped(self):
        # Raw pre-calibration values are engine-produced sigmoids and live
        # in (0,1); clamp defensively anyway (a>1 must never surface).
        d = self._draft()
        a = d.survival_confidences(
            DsparkDraftResult(tokens=[1, 2], confidences=[1.5, -0.5]))
        assert a == [1.0, 0.0]

    def test_combiner_uses_per_position_survival(self):
        # Per-position a_j replaces the flat proxy: identical latency and
        # cost knobs, so p_accept must equal a_j * coverage_discount.
        c = DraftCombiner()
        a = [0.9, 0.45, 0.09]
        combined = c.combine(
            prompt_lookup_tokens=[],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=3,
            max_verifiable_depth=3,
            dspark_result=DsparkDraftResult(tokens=[100, 101, 102],
                                            confidences=[0.9, 0.5, 0.2]),
            dspark_confidence=0.5,       # would be flat without DSP-6
            dspark_confidences=a,
        )
        assert combined.tokens == [100, 101, 102]
        got = [s.p_accept for s in combined.steps]
        assert got == a  # coverage_discount == 1.0 by default

    def test_combiner_falls_back_to_proxy_without_confidences(self):
        c = DraftCombiner()
        combined = c.combine(
            prompt_lookup_tokens=[],
            mtp_result=None,
            self_spec_result=None,
            prompt_lookup_acceptance_rate=0.0,
            max_depth=2,
            max_verifiable_depth=2,
            dspark_result=DsparkDraftResult(tokens=[100, 101]),
            dspark_confidence=0.7,
            dspark_confidences=None,
        )
        assert [s.p_accept for s in combined.steps] == [0.7, 0.7]
