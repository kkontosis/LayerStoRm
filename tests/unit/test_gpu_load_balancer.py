"""Tests for orchestrator.gpu_load_balancer — 7-term GPU scoring."""

from collections import Counter

import pytest

from orchestrator.gpu_load_balancer import (
    GpuLoadBalancer,
    GpuLoadBalancerConfig,
    LearnedGpuScorer,
    LearnedGpuScorerConfig,
    _compute_capacity,
    _compute_residency,
)
from orchestrator.scheduler import GpuAssigner
from orchestrator.types import (
    ExpertKey,
    GpuConfig,
    WorkItem,
    WorkOperation,
    WorkStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_configs() -> tuple[GpuConfig, ...]:
    return (
        GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                  vram_bytes=32 * 1024**3, compute_weight=1.0),
        GpuConfig(position=1, gpu_type="rtx5090", is_tp=True,
                  vram_bytes=32 * 1024**3, compute_weight=1.0),
        GpuConfig(position=2, gpu_type="rtx5080", is_tp=False,
                  vram_bytes=16 * 1024**3, compute_weight=0.5),
        GpuConfig(position=3, gpu_type="rtx5080", is_tp=False,
                  vram_bytes=16 * 1024**3, compute_weight=0.5),
    )


def _cfg(**overrides) -> GpuLoadBalancerConfig:
    defaults: dict = {"gpus": _gpu_configs()}
    defaults.update(overrides)
    return GpuLoadBalancerConfig(**defaults)


def _balancer(**overrides) -> GpuLoadBalancer:
    return GpuLoadBalancer(_cfg(**overrides))


def _item(
    op: WorkOperation = WorkOperation.EXPERT_FFN,
    experts: list[ExpertKey] | None = None,
    target_gpu: int = 0,
    priority: float = 0.0,
    request_id: int = 1,
    layer: int = 3,
) -> WorkItem:
    return WorkItem(
        request_id=request_id,
        layer_idx=layer,
        operation=op,
        target_gpu=target_gpu,
        status=WorkStatus.READY,
        required_experts=experts or [],
        priority=priority,
    )


def _empty_state(gpus: tuple[GpuConfig, ...] | None = None):
    gpus = gpus or _gpu_configs()
    return {
        "resident_keys_per_gpu": {g.position: set() for g in gpus},
        "queue_depths": {g.position: 0 for g in gpus},
        "vram_free": {g.position: g.vram_bytes for g in gpus},
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        b = _balancer()
        assert set(b._gpu_indices) == {0, 1, 2, 3}
        assert set(b._tp_gpu_indices) == {0, 1}

    def test_empty_gpus_raises(self):
        with pytest.raises(ValueError, match="gpus must not be empty"):
            GpuLoadBalancer(GpuLoadBalancerConfig(gpus=()))

    def test_single_gpu(self):
        single = GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                           vram_bytes=32 * 1024**3, compute_weight=1.0)
        b = GpuLoadBalancer(GpuLoadBalancerConfig(gpus=(single,)))
        state = _empty_state((single,))
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        assert b.assign(item) == 0
        assert b.assign(_item(op=WorkOperation.ATTENTION)) == 0


# ---------------------------------------------------------------------------
# Asymmetric GPU balancing
# ---------------------------------------------------------------------------

class TestAsymmetricGpuBalancing:
    def test_5090_gets_more_work(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = b.assign(item)
        assert gpu in (0, 1)

    def test_capacity_score_asymmetry(self):
        g5090 = GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                          vram_bytes=32 * 1024**3, compute_weight=1.0)
        g5080 = GpuConfig(position=2, gpu_type="rtx5080", is_tp=False,
                          vram_bytes=16 * 1024**3, compute_weight=0.5)
        assert _compute_capacity(g5090, 2) > _compute_capacity(g5080, 2)

    def test_batch_distributes_proportionally(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        items = [_item(request_id=i, experts=[ExpertKey(3, i)])
                 for i in range(12)]
        b.assign_batch(items, **state)
        counts = Counter(it.target_gpu for it in items)
        tp_total = counts.get(0, 0) + counts.get(1, 0)
        non_tp_total = counts.get(2, 0) + counts.get(3, 0)
        assert tp_total > non_tp_total

    def test_equal_queue_prefers_higher_compute(self):
        b = _balancer()
        state = _empty_state()
        state["queue_depths"] = {0: 3, 1: 3, 2: 3, 3: 3}
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = b.assign(item)
        assert gpu in (0, 1)


# ---------------------------------------------------------------------------
# Residency preference
# ---------------------------------------------------------------------------

class TestResidencyPreference:
    def test_resident_gpu_wins(self):
        b = _balancer(weight_residency=0.9, weight_capacity=0.01,
                      weight_headroom=0.01, weight_affinity=0.01,
                      weight_transfer_cost=0.01,
                      weight_cache_quality_stable=0.01,
                      weight_cache_quality_streaming=0.01)
        state = _empty_state()
        ek = ExpertKey(3, 10)
        state["resident_keys_per_gpu"][2] = {ek}
        b.update_state(**state)
        item = _item(experts=[ek])
        assert b.assign(item) == 2

    def test_partial_residency_prefers_more_overlap(self):
        b = _balancer(weight_residency=0.9, weight_capacity=0.01,
                      weight_headroom=0.01, weight_affinity=0.01,
                      weight_transfer_cost=0.01,
                      weight_cache_quality_stable=0.01,
                      weight_cache_quality_streaming=0.01)
        experts = [ExpertKey(3, i) for i in range(4)]
        state = _empty_state()
        state["resident_keys_per_gpu"][0] = set(experts[:3])
        state["resident_keys_per_gpu"][2] = {experts[3]}
        b.update_state(**state)
        item = _item(experts=experts)
        assert b.assign(item) == 0

    def test_full_residency_beats_capacity(self):
        b = _balancer()
        experts = [ExpertKey(3, i) for i in range(4)]
        state = _empty_state()
        state["resident_keys_per_gpu"][2] = set(experts)
        b.update_state(**state)
        item = _item(experts=experts)
        assert b.assign(item) == 2

    def test_no_experts_equal_residency(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        item = _item(op=WorkOperation.GATING, experts=[])
        gpu = b.assign(item)
        assert gpu in (0, 1)


# ---------------------------------------------------------------------------
# Queue depth avoidance
# ---------------------------------------------------------------------------

class TestQueueDepthAvoidance:
    def test_high_queue_depth_avoided(self):
        b = _balancer()
        state = _empty_state()
        state["queue_depths"] = {0: 50, 1: 50, 2: 0, 3: 0}
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = b.assign(item)
        assert gpu in (2, 3)

    def test_incremental_queue_in_batch(self):
        b = _balancer()
        state = _empty_state()
        items = [_item(request_id=i, experts=[ExpertKey(3, i)])
                 for i in range(8)]
        b.assign_batch(items, **state)
        counts = Counter(it.target_gpu for it in items)
        assert max(counts.values()) <= 4

    def test_zero_queue_equal_capacity(self):
        g = GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3, compute_weight=1.0)
        assert _compute_capacity(g, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Batch distribution
# ---------------------------------------------------------------------------

class TestBatchDistribution:
    def test_batch_sets_target_gpu(self):
        b = _balancer()
        state = _empty_state()
        items = [_item(request_id=i) for i in range(5)]
        result = b.assign_batch(items, **state)
        assert all(it.target_gpu in {0, 1, 2, 3} for it in result)

    def test_batch_empty(self):
        b = _balancer()
        state = _empty_state()
        assert b.assign_batch([], **state) == []

    def test_batch_mixed_ops(self):
        b = _balancer()
        state = _empty_state()
        items = [
            _item(op=WorkOperation.ATTENTION, request_id=1),
            _item(op=WorkOperation.EXPERT_FFN, request_id=2,
                  experts=[ExpertKey(3, 0)]),
            _item(op=WorkOperation.EMBEDDING, request_id=3),
        ]
        b.assign_batch(items, **state)
        assert items[0].target_gpu in (0, 1)
        assert items[2].target_gpu in (0, 1)


# ---------------------------------------------------------------------------
# TP pinning
# ---------------------------------------------------------------------------

class TestTpPinning:
    def test_attention_to_tp(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        for _ in range(10):
            item = _item(op=WorkOperation.ATTENTION)
            assert b.assign(item) in (0, 1)

    def test_embedding_output_head_to_tp(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        for op in (WorkOperation.EMBEDDING, WorkOperation.OUTPUT_HEAD):
            item = _item(op=op)
            assert b.assign(item) in (0, 1)


# ---------------------------------------------------------------------------
# Transfer cost & urgency
# ---------------------------------------------------------------------------

class TestTransferCostUrgency:
    def test_urgent_item_avoids_congested_gpu(self):
        b = _balancer(
            weight_transfer_cost=0.8,
            weight_residency=0.01, weight_capacity=0.01,
            weight_headroom=0.01, weight_affinity=0.01,
            weight_cache_quality_stable=0.01,
            weight_cache_quality_streaming=0.01,
        )
        state = _empty_state()
        state["pending_bin_counts_per_gpu"] = {
            0: [0, 0, 10], 1: [0, 0, 10],
            2: [0, 0, 0], 3: [0, 0, 0],
        }
        b.update_state(**state)
        urgent = _item(experts=[ExpertKey(3, 0)], priority=0.9)
        gpu = b.assign(urgent)
        assert gpu in (2, 3)

    def test_pending_bins_incremented_in_batch(self):
        b = _balancer()
        state = _empty_state()
        state["pending_bin_counts_per_gpu"] = {
            0: [0, 0, 0], 1: [0, 0, 0],
            2: [0, 0, 0], 3: [0, 0, 0],
        }
        items = [_item(request_id=i, experts=[ExpertKey(3, 100 + i)],
                       priority=0.5) for i in range(4)]
        b.assign_batch(items, **state)
        counts = Counter(it.target_gpu for it in items)
        assert max(counts.values()) <= 2

    def test_congestion_from_higher_bins(self):
        b = _balancer(
            weight_transfer_cost=0.9,
            weight_residency=0.01, weight_capacity=0.01,
            weight_headroom=0.01, weight_affinity=0.01,
            weight_cache_quality_stable=0.01,
            weight_cache_quality_streaming=0.01,
        )
        state = _empty_state()
        state["pending_bin_counts_per_gpu"] = {
            0: [10, 0, 0], 1: [10, 0, 0],
            2: [0, 0, 0], 3: [0, 0, 0],
        }
        b.update_state(**state)
        mid = _item(experts=[ExpertKey(3, 0)], priority=0.5)
        gpu = b.assign(mid)
        assert gpu in (0, 1, 2, 3)

        state["pending_bin_counts_per_gpu"] = {
            0: [0, 10, 0], 1: [0, 10, 0],
            2: [0, 0, 0], 3: [0, 0, 0],
        }
        b.update_state(**state)
        gpu2 = b.assign(mid)
        assert gpu2 in (2, 3)


# ---------------------------------------------------------------------------
# Cache quality
# ---------------------------------------------------------------------------

class TestCacheQuality:
    def test_low_quality_cache_preferred(self):
        b = _balancer(
            weight_cache_quality_stable=0.8,
            weight_residency=0.01, weight_capacity=0.01,
            weight_headroom=0.01, weight_affinity=0.01,
            weight_transfer_cost=0.01,
            weight_cache_quality_streaming=0.01,
        )
        state = _empty_state()
        state["cache_quality_stable"] = {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.9}
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = b.assign(item)
        assert gpu in (2, 3)

    def test_cache_quality_zero_when_empty(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        item = _item(experts=[ExpertKey(3, 0)])
        b.assign(item)


# ---------------------------------------------------------------------------
# Recommend transfer target
# ---------------------------------------------------------------------------

class TestRecommendTransferTarget:
    def test_affinity_preferred(self):
        b = _balancer()
        state = _empty_state()
        affinity = {ExpertKey(3, 5): 2}
        gpu = b.recommend_transfer_target(
            ExpertKey(3, 5), **state, affinity_map=affinity,
        )
        assert gpu == 2

    def test_headroom_tiebreaker(self):
        b = _balancer(
            weight_headroom=0.8,
            weight_residency=0.01, weight_capacity=0.01,
            weight_affinity=0.01,
            weight_cache_quality_stable=0.01,
            weight_cache_quality_streaming=0.01,
        )
        state = _empty_state()
        state["vram_free"] = {
            0: 1 * 1024**3, 1: 1 * 1024**3,
            2: 15 * 1024**3, 3: 15 * 1024**3,
        }
        gpu = b.recommend_transfer_target(ExpertKey(3, 5), **state)
        assert gpu in (2, 3)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_implements_gpu_assigner(self):
        b = _balancer()
        state = _empty_state()
        b.update_state(**state)
        assert isinstance(b, GpuAssigner)

    def test_learned_scorer_implements_gpu_assigner(self):
        fb = _balancer()
        state = _empty_state()
        fb.update_state(**state)
        s = LearnedGpuScorer(LearnedGpuScorerConfig(enabled=True), fb)
        assert isinstance(s, GpuAssigner)


# ---------------------------------------------------------------------------
# Learned scorer — construction
# ---------------------------------------------------------------------------

def _learned_scorer(enabled: bool = True, min_samples: int = 4,
                    **fb_overrides) -> LearnedGpuScorer:
    fb = _balancer(**fb_overrides)
    fb.update_state(**_empty_state())
    return LearnedGpuScorer(
        LearnedGpuScorerConfig(enabled=enabled,
                               min_samples_before_predict=min_samples,
                               training_batch_size=4),
        fb,
    )


class TestLearnedScorerConstruction:
    def test_construction(self):
        s = _learned_scorer()
        assert not s.is_warm()
        assert s.sample_count() == 0

    def test_disabled_delegates_to_fallback(self):
        s = _learned_scorer(enabled=False)
        # Seed buffer to make it technically warm
        import numpy as np
        for i in range(10):
            s._buffer.add(np.zeros(7, dtype=np.float32),
                          np.array([1.0], dtype=np.float32))
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = s.assign(item)
        assert gpu in {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Learned scorer — cold start / fallback
# ---------------------------------------------------------------------------

class TestLearnedScorerFallback:
    def test_cold_start_uses_heuristic(self):
        s = _learned_scorer(min_samples=100)
        item = _item(experts=[ExpertKey(3, 0)])
        gpu = s.assign(item)
        assert gpu in {0, 1, 2, 3}

    def test_tp_ops_always_heuristic(self):
        s = _learned_scorer()
        import numpy as np
        for i in range(10):
            s._buffer.add(np.zeros(7, dtype=np.float32),
                          np.array([1.0], dtype=np.float32))
        item = _item(op=WorkOperation.ATTENTION)
        assert s.assign(item) in (0, 1)

    def test_warm_after_min_samples(self):
        s = _learned_scorer(min_samples=4)
        import numpy as np
        assert not s.is_warm()
        for i in range(4):
            s._buffer.add(np.zeros(7, dtype=np.float32),
                          np.array([1.0], dtype=np.float32))
        assert s.is_warm()


# ---------------------------------------------------------------------------
# Learned scorer — training and prediction
# ---------------------------------------------------------------------------

class TestLearnedScorerTraining:
    def test_record_outcome_stores_sample(self):
        import numpy as np
        s = _learned_scorer(min_samples=4)
        for i in range(4):
            s._buffer.add(np.zeros(7, dtype=np.float32),
                          np.array([1.0], dtype=np.float32))
        assert s.is_warm()
        item = _item(request_id=42, experts=[ExpertKey(3, 0)])
        gpu = s.assign(item)
        count_before = s.sample_count()
        s.record_outcome(42, gpu, 100.0)
        assert s.sample_count() == count_before + 1

    def test_maybe_train_updates_weights(self):
        import numpy as np
        s = _learned_scorer(min_samples=4)
        w_before = s._weights[s._active_buf]["w1"].copy()
        for i in range(8):
            s._buffer.add(
                np.random.default_rng(i).standard_normal(7).astype(np.float32),
                np.array([float(i)], dtype=np.float32),
            )
        assert s.maybe_train()
        assert s._train_count == 1
        w_after = s._weights[s._active_buf]["w1"]
        assert not np.allclose(w_before, w_after)

    def test_predict_latency_forward_pass(self):
        import numpy as np
        s = _learned_scorer()
        w = s._weights[s._active_buf]
        features = np.ones(7, dtype=np.float32)
        hidden = np.maximum(0.0, features @ w["w1"] + w["b1"])
        expected = float((hidden @ w["w2"] + w["b2"])[0])
        actual = s._predict_latency(features)
        assert actual == pytest.approx(expected)

    def test_learned_scorer_picks_lower_latency(self):
        import numpy as np
        s = _learned_scorer(min_samples=4)
        rng = np.random.default_rng(99)
        for i in range(200):
            features = rng.standard_normal(7).astype(np.float32)
            features[1] = 0.9  # high capacity
            latency = 50.0 + rng.standard_normal() * 5
            s._buffer.add(features, np.array([latency], dtype=np.float32))
        for i in range(200):
            features = rng.standard_normal(7).astype(np.float32)
            features[1] = 0.1  # low capacity
            latency = 200.0 + rng.standard_normal() * 5
            s._buffer.add(features, np.array([latency], dtype=np.float32))
        for _ in range(50):
            s.maybe_train()
        high_cap = np.array([0.5, 0.9, 0.5, 0.5, 0.1, 0.5, 0.5],
                            dtype=np.float32)
        low_cap = np.array([0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5],
                           dtype=np.float32)
        assert s._predict_latency(high_cap) < s._predict_latency(low_cap)

    def test_double_buffer_swap(self):
        import numpy as np
        s = _learned_scorer(min_samples=4)
        initial_active = s._active_buf
        for i in range(8):
            s._buffer.add(
                np.random.default_rng(i).standard_normal(7).astype(np.float32),
                np.array([float(i)], dtype=np.float32),
            )
        s.maybe_train()
        assert s._active_buf != initial_active
