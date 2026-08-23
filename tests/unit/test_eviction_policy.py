"""Tests for orchestrator.eviction_policy — scoring, ranking, selection, hysteresis."""

import pytest

from orchestrator.types import (
    CacheZone,
    EvictionPlanEntry,
    EvictionPolicyType,
    ExpertEvictionInput,
    ExpertKey,
)
from orchestrator.eviction_policy import (
    DUPLICATE_PENALTY,
    STREAMING_ZONE_BONUS,
    EvictionPolicy,
    EvictionPolicyConfig,
    HysteresisTracker,
)
from orchestrator.scheduler import EvictionScorer


def _inp(layer=0, expert=0, zone=CacheZone.STREAMING, is_dup=False, gpu=0,
         rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=0.0, ps=0.0, hs=0.0):
    return ExpertEvictionInput(
        key=ExpertKey(layer, expert), zone=zone, is_duplicate=is_dup,
        gpu_idx=gpu, recency=rec, frequency=freq, routing_weight=rw,
        temporal_autocorr=ta, coactivation=co, prefetch_score=ps,
        hysteresis_state=hs)


def _default_cfg():
    return EvictionPolicyConfig()


def _lru_cfg():
    return EvictionPolicyConfig(policy=EvictionPolicyType.LRU)


def _lfu_cfg():
    return EvictionPolicyConfig(policy=EvictionPolicyType.LFU)


def _all_weights_cfg():
    return EvictionPolicyConfig(
        alpha_recency=0.3,
        beta_frequency=0.2,
        gamma_routing_weight=0.15,
        delta_temporal_autocorr=0.15,
        epsilon_coactivation=0.1,
        zeta_prefetch_score=0.1,
    )


# ── Construction tests ──────────────────────────────────────────────────────


class TestConstruction:
    def test_default_config(self):
        ep = EvictionPolicy(_default_cfg())
        cfg = ep.config
        assert cfg.policy == EvictionPolicyType.IMPACT_WEIGHTED_LRU
        assert cfg.alpha_recency == 0.4
        assert cfg.beta_frequency == 0.35
        assert cfg.gamma_routing_weight == 0.25
        assert cfg.delta_temporal_autocorr == 0.0
        assert cfg.epsilon_coactivation == 0.0
        assert cfg.zeta_prefetch_score == 0.0
        assert cfg.eta_hysteresis == 0.0

    def test_lru_policy(self):
        ep = EvictionPolicy(_lru_cfg())
        assert ep.config.policy == EvictionPolicyType.LRU

    def test_lfu_policy(self):
        ep = EvictionPolicy(_lfu_cfg())
        assert ep.config.policy == EvictionPolicyType.LFU

    def test_custom_weights(self):
        cfg = _all_weights_cfg()
        ep = EvictionPolicy(cfg)
        assert ep.config.alpha_recency == pytest.approx(0.3)
        assert ep.config.beta_frequency == pytest.approx(0.2)
        assert ep.config.gamma_routing_weight == pytest.approx(0.15)
        assert ep.config.delta_temporal_autocorr == pytest.approx(0.15)
        assert ep.config.epsilon_coactivation == pytest.approx(0.1)
        assert ep.config.zeta_prefetch_score == pytest.approx(0.1)


# ── Impact-weighted scoring tests ───────────────────────────────────────────


class TestImpactWeightedScoring:
    def test_score_all_zeros(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=0.0, freq=0.0, rw=0.0, ta=0.0, co=0.0, ps=0.0)
        assert ep.score(inp) == pytest.approx(0.0, abs=1e-12)

    def test_score_all_ones(self):
        ep = EvictionPolicy(_all_weights_cfg())
        inp = _inp(rec=1.0, freq=1.0, rw=1.0, ta=1.0, co=1.0, ps=1.0)
        # 0.3 - 0.2 - 0.15 - 0.15 - 0.1 - 0.1 = -0.4
        assert ep.score(inp) == pytest.approx(-0.4, abs=1e-12)

    def test_score_recency_dominates(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=1.0, freq=0.0, rw=0.0)
        assert ep.score(inp) == pytest.approx(0.4, abs=1e-12)

    def test_score_frequency_protects(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=0.0, freq=1.0, rw=0.0)
        assert ep.score(inp) == pytest.approx(-0.35, abs=1e-12)

    def test_score_routing_weight_protects(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=0.0, freq=0.0, rw=1.0)
        assert ep.score(inp) == pytest.approx(-0.25, abs=1e-12)

    def test_score_temporal_autocorr_protects(self):
        ep = EvictionPolicy(_all_weights_cfg())
        base = _inp(rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=0.0, ps=0.0)
        with_ta = _inp(rec=0.5, freq=0.5, rw=0.5, ta=1.0, co=0.0, ps=0.0)
        assert ep.score(with_ta) < ep.score(base)

    def test_score_coactivation_protects(self):
        ep = EvictionPolicy(_all_weights_cfg())
        base = _inp(rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=0.0, ps=0.0)
        with_co = _inp(rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=1.0, ps=0.0)
        assert ep.score(with_co) < ep.score(base)

    def test_score_prefetch_protects(self):
        ep = EvictionPolicy(_all_weights_cfg())
        base = _inp(rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=0.0, ps=0.0)
        with_ps = _inp(rec=0.5, freq=0.5, rw=0.5, ta=0.0, co=0.0, ps=1.0)
        assert ep.score(with_ps) < ep.score(base)

    def test_score_linearity(self):
        ep = EvictionPolicy(_default_cfg())
        in1 = _inp(rec=0.5, freq=0.0, rw=0.0)
        in2 = _inp(rec=1.0, freq=0.0, rw=0.0)
        assert ep.score(in2) == pytest.approx(ep.score(in1) * 2.0, abs=1e-12)


# ── Duplicate handling tests ────────────────────────────────────────────────


class TestDuplicateHandling:
    def test_duplicate_always_more_evictable(self):
        ep = EvictionPolicy(_default_cfg())
        dup = _inp(expert=0, is_dup=True, rec=0.0, freq=1.0, rw=1.0)
        primary = _inp(expert=1, is_dup=False, rec=1.0, freq=0.0, rw=0.0)
        assert ep.score(dup) > ep.score(primary)

    def test_duplicate_before_primary(self):
        ep = EvictionPolicy(_default_cfg())
        dup = _inp(is_dup=True)
        primary = _inp(is_dup=False)
        assert ep.score(dup) - ep.score(primary) == pytest.approx(
            DUPLICATE_PENALTY, abs=1e-12)

    def test_multiple_duplicates_sort_first(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, is_dup=False, rec=1.0, freq=0.0, rw=0.0),
            _inp(expert=1, is_dup=True, rec=0.0, freq=1.0, rw=1.0),
            _inp(expert=2, is_dup=False, rec=0.8, freq=0.0, rw=0.0),
            _inp(expert=3, is_dup=True, rec=0.0, freq=0.8, rw=0.8),
        ]
        ranked = ep.rank_candidates(inputs)
        assert len(ranked) == 4
        assert ranked[0].key.expert_idx == 1 or ranked[0].key.expert_idx == 3
        assert ranked[1].key.expert_idx == 1 or ranked[1].key.expert_idx == 3
        assert ranked[0].key.expert_idx != ranked[1].key.expert_idx
        # All duplicates before all primaries
        assert ranked[0].eviction_score > ranked[2].eviction_score
        assert ranked[1].eviction_score > ranked[2].eviction_score


# ── LRU policy tests ──���────────────────────────────────────────────────────


class TestLruPolicy:
    def test_lru_only_uses_recency(self):
        ep = EvictionPolicy(_lru_cfg())
        in1 = _inp(rec=0.5, freq=0.0, rw=0.0)
        in2 = _inp(rec=0.5, freq=1.0, rw=1.0)
        assert ep.score(in1) == pytest.approx(ep.score(in2), abs=1e-12)

    def test_lru_high_recency_more_evictable(self):
        ep = EvictionPolicy(_lru_cfg())
        old = _inp(rec=0.9, freq=0.5, rw=0.5)
        new = _inp(expert=1, rec=0.1, freq=0.5, rw=0.5)
        assert ep.score(old) > ep.score(new)

    def test_lru_duplicate_still_first(self):
        ep = EvictionPolicy(_lru_cfg())
        dup = _inp(is_dup=True, rec=0.0)
        primary = _inp(expert=1, is_dup=False, rec=1.0)
        assert ep.score(dup) > ep.score(primary)


# ── LFU policy tests ───────────────────────────────────────────────────────


class TestLfuPolicy:
    def test_lfu_only_uses_frequency(self):
        ep = EvictionPolicy(_lfu_cfg())
        in1 = _inp(rec=0.0, freq=0.5, rw=0.0)
        in2 = _inp(rec=1.0, freq=0.5, rw=1.0)
        assert ep.score(in1) == pytest.approx(ep.score(in2), abs=1e-12)

    def test_lfu_low_frequency_more_evictable(self):
        ep = EvictionPolicy(_lfu_cfg())
        low = _inp(freq=0.1, rw=0.5)
        high = _inp(expert=1, freq=0.9, rw=0.5)
        assert ep.score(low) > ep.score(high)

    def test_lfu_duplicate_still_first(self):
        ep = EvictionPolicy(_lfu_cfg())
        dup = _inp(is_dup=True, freq=1.0)
        primary = _inp(expert=1, is_dup=False, freq=0.0)
        assert ep.score(dup) > ep.score(primary)


# ── rank_candidates tests ──────────────────────────────────────────────────


class TestRankCandidates:
    def test_empty_input(self):
        ep = EvictionPolicy(_default_cfg())
        assert ep.rank_candidates([]) == []

    def test_single_item(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [_inp(layer=5, expert=42, zone=CacheZone.STABLE, gpu=2,
                       rec=0.7, freq=0.3, rw=0.2)]
        ranked = ep.rank_candidates(inputs)
        assert len(ranked) == 1
        assert ranked[0].key == ExpertKey(5, 42)
        assert ranked[0].zone == CacheZone.STABLE
        assert ranked[0].gpu_idx == 2

    def test_ordering(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, rec=0.1, freq=0.9, rw=0.9),
            _inp(expert=1, rec=0.9, freq=0.1, rw=0.1),
            _inp(expert=2, rec=0.5, freq=0.5, rw=0.5),
        ]
        ranked = ep.rank_candidates(inputs)
        assert len(ranked) == 3
        assert ranked[0].key.expert_idx == 1
        assert ranked[2].key.expert_idx == 0
        assert ranked[0].eviction_score >= ranked[1].eviction_score
        assert ranked[1].eviction_score >= ranked[2].eviction_score

    def test_duplicates_first(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, is_dup=False, rec=1.0, freq=0.0, rw=0.0),
            _inp(expert=1, is_dup=True, rec=0.0, freq=1.0, rw=1.0),
        ]
        ranked = ep.rank_candidates(inputs)
        assert len(ranked) == 2
        assert ranked[0].key.expert_idx == 1  # duplicate first


# ── select_evictions tests ─────────────────────────────────────────────────


class TestSelectEvictions:
    def test_filters_gpu(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, gpu=0, rec=0.9),
            _inp(expert=1, gpu=1, rec=0.8),
            _inp(expert=2, gpu=0, rec=0.7),
            _inp(expert=3, gpu=2, rec=0.6),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 10)
        assert len(result) == 2
        for c in result:
            assert c.gpu_idx == 0

    def test_streaming_zone_only(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, zone=CacheZone.STREAMING, rec=0.9),
            _inp(expert=1, zone=CacheZone.STABLE, rec=0.8),
            _inp(expert=2, zone=CacheZone.STREAMING, rec=0.7),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 10)
        assert len(result) == 2
        for c in result:
            assert c.zone == CacheZone.STREAMING

    def test_stable_zone_both_eligible(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, zone=CacheZone.STABLE, rec=1.0, freq=0.0, rw=0.0),
            _inp(expert=1, zone=CacheZone.STREAMING, rec=0.1, freq=0.9, rw=0.9),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STABLE, 2)
        assert len(result) == 2
        assert result[0].zone == CacheZone.STREAMING
        assert result[1].zone == CacheZone.STABLE

    def test_count_limit(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, rec=0.9),
            _inp(expert=1, rec=0.8),
            _inp(expert=2, rec=0.7),
            _inp(expert=3, rec=0.6),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 2)
        assert len(result) == 2
        assert result[0].key.expert_idx == 0
        assert result[1].key.expert_idx == 1

    def test_count_exceeds_available(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, rec=0.9),
            _inp(expert=1, rec=0.8),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 100)
        assert len(result) == 2

    def test_zero_count(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [_inp()]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 0)
        assert result == []

    def test_multi_gpu_isolation(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, gpu=0),
            _inp(expert=1, gpu=1),
            _inp(expert=2, gpu=2),
            _inp(expert=3, gpu=3),
        ]
        result = ep.select_evictions(inputs, 1, CacheZone.STREAMING, 10)
        assert len(result) == 1
        assert result[0].gpu_idx == 1
        assert result[0].key.expert_idx == 1

    def test_empty_inputs(self):
        ep = EvictionPolicy(_default_cfg())
        result = ep.select_evictions([], 0, CacheZone.STREAMING, 5)
        assert result == []

    def test_no_matching_gpu(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, gpu=0),
            _inp(expert=1, gpu=1),
        ]
        result = ep.select_evictions(inputs, 5, CacheZone.STREAMING, 10)
        assert result == []


# ── Zone + duplicate interaction tests ──────────────────────────────────────


class TestZoneDuplicateInteraction:
    def test_streaming_duplicate_before_stable_primary(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, zone=CacheZone.STABLE, is_dup=False,
                 rec=1.0, freq=0.0, rw=0.0),
            _inp(expert=1, zone=CacheZone.STREAMING, is_dup=True,
                 rec=0.0, freq=1.0, rw=1.0),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STABLE, 2)
        assert len(result) == 2
        assert result[0].key.expert_idx == 1  # streaming duplicate first
        assert result[0].zone == CacheZone.STREAMING

    def test_stable_duplicate_before_stable_primary(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [
            _inp(expert=0, zone=CacheZone.STABLE, is_dup=False,
                 rec=1.0, freq=0.0, rw=0.0),
            _inp(expert=1, zone=CacheZone.STABLE, is_dup=True,
                 rec=0.0, freq=1.0, rw=1.0),
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STABLE, 2)
        assert len(result) == 2
        assert result[0].key.expert_idx == 1  # duplicate first


# ── Edge case tests ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_negative_score_valid(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=0.0, freq=1.0, rw=1.0)
        assert ep.score(inp) < 0.0

    def test_all_same_score_deterministic(self):
        ep = EvictionPolicy(_default_cfg())
        inputs = [_inp(expert=i) for i in range(10)]
        ranked = ep.rank_candidates(inputs)
        assert len(ranked) == 10
        for i in range(1, len(ranked)):
            assert ranked[i].eviction_score == pytest.approx(
                ranked[0].eviction_score, abs=1e-12)


# ── Protocol conformance ───────────────────────────────────────────────────


class TestProtocol:
    def test_implements_eviction_scorer(self):
        ep = EvictionPolicy()
        assert isinstance(ep, EvictionScorer)

    def test_duplicate_penalty_value(self):
        assert DUPLICATE_PENALTY == 1000.0

    def test_streaming_zone_bonus_value(self):
        assert STREAMING_ZONE_BONUS == 500.0


# ── Hysteresis tracker tests ��─────────────────────────────────────────────


class TestHysteresisTracker:
    def _cfg(self, stages=5, up=5, down=1):
        return EvictionPolicyConfig(
            hysteresis_enabled=True,
            hysteresis_stages=stages,
            hysteresis_up_threshold=up,
            hysteresis_down_threshold=down,
        )

    def test_initial_state_zero(self):
        t = HysteresisTracker(self._cfg())
        assert t.state(ExpertKey(0, 0)) == 0

    def test_promote(self):
        t = HysteresisTracker(self._cfg())
        key = ExpertKey(0, 0)
        # activation=5, state=0, diff=5 >= up_threshold=5 → promote
        new_state = t.update(key, 5)
        assert new_state == 1

    def test_no_change_in_band(self):
        t = HysteresisTracker(self._cfg())
        key = ExpertKey(0, 0)
        # activation=3, state=0, diff=3 < up_threshold=5 → no change
        new_state = t.update(key, 3)
        assert new_state == 0

    def test_demote(self):
        t = HysteresisTracker(self._cfg())
        key = ExpertKey(0, 0)
        # Build up to state 3
        for _ in range(3):
            t.update(key, t.state(key) + 5)
        assert t.state(key) == 3
        # activation=1, state=3, diff=-2, |-2| >= down_threshold=1 → demote
        new_state = t.update(key, 1)
        assert new_state == 2

    def test_state_capped(self):
        t = HysteresisTracker(self._cfg(stages=3))
        key = ExpertKey(0, 0)
        for _ in range(10):
            t.update(key, t.state(key) + 5)
        assert t.state(key) == 3

    def test_state_floored(self):
        t = HysteresisTracker(self._cfg())
        key = ExpertKey(0, 0)
        # Already at 0, try to demote further
        new_state = t.update(key, -10)
        assert new_state == 0

    def test_normalized_state(self):
        t = HysteresisTracker(self._cfg(stages=5))
        key = ExpertKey(0, 0)
        assert t.normalized_state(key) == pytest.approx(0.0)
        # Promote to 1
        t.update(key, 5)
        assert t.normalized_state(key) == pytest.approx(0.2)
        # Promote to 2
        t.update(key, 6 + 1)  # state=1, need diff >= 5 → activation >= 6
        assert t.normalized_state(key) == pytest.approx(0.4)

    def test_reset(self):
        t = HysteresisTracker(self._cfg())
        key = ExpertKey(0, 0)
        t.update(key, 5)
        assert t.state(key) == 1
        t.reset()
        assert t.state(key) == 0


# ── Hysteresis scoring tests ───────────────────────────────────────────────


class TestHysteresisScoring:
    def test_eta_zero_matches_cpp(self):
        ep = EvictionPolicy(_default_cfg())
        inp = _inp(rec=1.0, freq=0.0, rw=0.0, hs=1.0)
        # eta=0 so hysteresis_state has no effect
        assert ep.score(inp) == pytest.approx(0.4, abs=1e-12)

    def test_eta_nonzero_protects(self):
        cfg = EvictionPolicyConfig(eta_hysteresis=0.3)
        ep = EvictionPolicy(cfg)
        base = _inp(rec=0.5, freq=0.5, rw=0.5, hs=0.0)
        with_hs = _inp(rec=0.5, freq=0.5, rw=0.5, hs=1.0)
        assert ep.score(with_hs) < ep.score(base)
        diff = ep.score(base) - ep.score(with_hs)
        assert diff == pytest.approx(0.3, abs=1e-12)

    def test_hysteresis_prevents_eviction(self):
        cfg = EvictionPolicyConfig(eta_hysteresis=0.3)
        ep = EvictionPolicy(cfg)
        inputs = [
            _inp(expert=0, rec=0.8, freq=0.2, rw=0.2, hs=0.8),  # protected
            _inp(expert=1, rec=0.8, freq=0.2, rw=0.2, hs=0.0),  # unprotected
        ]
        result = ep.select_evictions(inputs, 0, CacheZone.STREAMING, 1)
        assert len(result) == 1
        assert result[0].key.expert_idx == 1  # unprotected evicts first
