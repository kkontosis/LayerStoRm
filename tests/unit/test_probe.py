"""Tests for orchestrator.probe — multi-layer predictive prefetching."""

import numpy as np
import pytest

from orchestrator.types import (
    EngineMetadata,
    ExpertKey,
    GpuConfig,
    PrefetchConfidence,
    PrefetchSource,
)
from orchestrator.probe import (
    Probe,
    ProbeConfig,
    ProbePredictor,
    ProbePredictorConfig,
)


def _meta(num_layers: int = 61, num_moe_layers: int = 58,
          num_experts: int = 256):
    return EngineMetadata(
        num_gpus=4, num_moe_layers=num_moe_layers,
        num_experts=num_experts, num_layers=num_layers,
        expert_bytes=2_359_296, kv_bytes_per_page=644 * 64,
        gpus=(
            GpuConfig(position=0, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3),
            GpuConfig(position=1, gpu_type="rtx5090", is_tp=True,
                      vram_bytes=32 * 1024**3),
            GpuConfig(position=2, gpu_type="rtx5080", is_tp=False,
                      vram_bytes=16 * 1024**3),
            GpuConfig(position=3, gpu_type="rtx5080", is_tp=False,
                      vram_bytes=16 * 1024**3),
        ),
    )


def _small_meta():
    """Small model for predictor tests (fewer experts = faster)."""
    return EngineMetadata(
        num_gpus=1, num_moe_layers=8, num_experts=16,
        num_layers=10, expert_bytes=1000, kv_bytes_per_page=100,
        gpus=(GpuConfig(position=0, gpu_type="test", is_tp=False,
                        vram_bytes=1024**3),),
    )


def _small_predictor_config(**overrides):
    defaults = dict(
        hidden_size=64, feature_dim=16,
        learning_rate=0.01, momentum=0.0,
        training_batch_size=8, training_buffer_size=128,
        min_samples_before_predict=8,
    )
    defaults.update(overrides)
    return ProbePredictorConfig(**defaults)


# ── Probe layer identification ────────────────────────────────────────────


class TestProbeLayerIndices:
    def test_default_probe_points(self):
        p = Probe(_meta())
        assert p.probe_layers == (15, 30, 45)

    def test_is_probe_layer(self):
        p = Probe(_meta())
        assert p.is_probe_layer(15)
        assert p.is_probe_layer(30)
        assert p.is_probe_layer(45)
        assert not p.is_probe_layer(0)
        assert not p.is_probe_layer(16)
        assert not p.is_probe_layer(60)

    def test_is_moe_layer(self):
        p = Probe(_meta())
        assert not p.is_moe_layer(0)
        assert not p.is_moe_layer(2)
        assert p.is_moe_layer(3)
        assert p.is_moe_layer(30)
        assert p.is_moe_layer(60)
        assert not p.is_moe_layer(61)

    def test_probe_layers_property(self):
        p = Probe(_meta(), ProbeConfig(probe_points=(0.5,)))
        assert p.probe_layers == (30,)

    def test_custom_probe_points(self):
        p = Probe(_meta(), ProbeConfig(probe_points=(0.1, 0.9)))
        assert p.probe_layers == (6, 54)

    def test_small_model_probe_points(self):
        p = Probe(_small_meta())
        assert p.probe_layers == (2, 4, 7)


# ── Config ────────────────────────────────────────────────────────────────


class TestProbeConfig:
    def test_default_config(self):
        p = Probe(_meta())
        assert p.config.enabled is True
        assert p.config.probe_points == (0.25, 0.5, 0.75)
        assert p.config.confidence_threshold == pytest.approx(0.6)
        assert p.config.top_k == 8
        assert p.config.score_threshold == pytest.approx(0.01)

    def test_custom_config(self):
        cfg = ProbeConfig(enabled=True, top_k=4, score_threshold=0.05,
                          confidence_threshold=0.8)
        p = Probe(_meta(), cfg)
        assert p.config.top_k == 4
        assert p.config.score_threshold == pytest.approx(0.05)

    def test_disabled_probe(self):
        p = Probe(_meta(), ProbeConfig(enabled=False))
        assert not p.enabled


# ── Predictor core ────────────────────────────────────────────────────────


class TestPredictor:
    def test_predict_cold_start_returns_empty(self):
        pred = ProbePredictor(_small_meta(), _small_predictor_config())
        h = np.random.randn(64).astype(np.float32)
        out = pred.predict(2, h)
        assert out == {}

    def test_predict_output_shape(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg)
        rng = np.random.default_rng(99)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)
        out = pred.predict(pp, h)
        assert set(out.keys()) == set(targets)
        for scores in out.values():
            assert scores.shape == (16,)

    def test_predict_output_range(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg)
        rng = np.random.default_rng(99)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        for _ in range(3):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(pp, h, t, mask)
        h_test = rng.standard_normal(64).astype(np.float32)
        out = pred.predict(pp, h_test)
        for scores in out.values():
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

    def test_predict_returns_all_target_layers(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg)
        pp = pred.probe_layer_indices[1]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)
        out = pred.predict(pp, h)
        assert len(out) == len(targets)

    def test_record_sample_increments_count(self):
        pred = ProbePredictor(_small_meta(), _small_predictor_config())
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        assert pred.sample_count(pp) == 0
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for t in targets:
            pred.record_sample(pp, h, t, mask)
        assert pred.sample_count(pp) == 1
        for t in targets:
            pred.record_sample(pp, h, t, mask)
        assert pred.sample_count(pp) == 2

    def test_ring_buffer_wraps(self):
        cfg = _small_predictor_config(training_buffer_size=4)
        pred = ProbePredictor(_small_meta(), cfg)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for _ in range(10):
            for t in targets:
                pred.record_sample(pp, h, t, mask)
        assert pred.sample_count(pp) == 4

    def test_maybe_train_updates_weights(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = ProbePredictor(meta, cfg)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(123)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(pp, h, t, mask)
        w_before = pred._weights[pred._active_buf[pp]][pp]['w_out'].copy()
        trained = pred.maybe_train(pp)
        assert trained
        w_after = pred._weights[pred._active_buf[pp]][pp]['w_out']
        assert not np.allclose(w_before, w_after)

    def test_double_buffer_swap(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = ProbePredictor(meta, cfg)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(456)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(pp, h, t, mask)
        buf_before = pred._active_buf[pp]
        pred.maybe_train(pp)
        assert pred._active_buf[pp] != buf_before

    def test_maybe_train_returns_false_insufficient_data(self):
        pred = ProbePredictor(_small_meta(), _small_predictor_config())
        pp = pred.probe_layer_indices[0]
        assert not pred.maybe_train(pp)

    def test_probe_point_layer_indices(self):
        pred = ProbePredictor(_small_meta(), _small_predictor_config())
        assert pred.probe_layer_indices == (2, 4, 7)

    def test_target_layers_only_after_probe_point(self):
        pred = ProbePredictor(_small_meta(), _small_predictor_config())
        for pp in pred.probe_layer_indices:
            for t in pred.target_layers_for(pp):
                assert t > pp

    def test_target_layers_are_moe_layers(self):
        meta = _small_meta()
        first_moe = meta.num_layers - meta.num_moe_layers
        pred = ProbePredictor(meta, _small_predictor_config())
        for pp in pred.probe_layer_indices:
            for t in pred.target_layers_for(pp):
                assert first_moe <= t < meta.num_layers


# ── Predictor training ────────────────────────────────────────────────────


class TestPredictorTraining:
    def test_predict_after_training_improves(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=8, min_samples_before_predict=1,
            learning_rate=0.05, momentum=0.0)
        pred = ProbePredictor(meta, cfg)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        target = targets[0]
        rng = np.random.default_rng(42)

        target_mask = np.zeros(16, dtype=np.float32)
        target_mask[0] = 1.0
        target_mask[1] = 1.0

        h_test = rng.standard_normal(64).astype(np.float32)
        for _ in range(50):
            h = h_test + rng.standard_normal(64).astype(np.float32) * 0.1
            for t in targets:
                pred.record_sample(pp, h, t, target_mask)

        pred_before = pred.predict(pp, h_test)
        score_before = pred_before[target][0] + pred_before[target][1]
        for _ in range(30):
            pred.maybe_train(pp)
        pred_after = pred.predict(pp, h_test)
        score_after = pred_after[target][0] + pred_after[target][1]
        assert score_after > score_before

    def test_training_independent_per_probe_point(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = ProbePredictor(meta, cfg)
        pp0 = pred.probe_layer_indices[0]
        pp1 = pred.probe_layer_indices[1]
        rng = np.random.default_rng(77)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in pred.target_layers_for(pp0):
                pred.record_sample(pp0, h, t, mask)
        w1_before = pred._weights[pred._active_buf[pp1]][pp1]['w_out'].copy()
        pred.maybe_train(pp0)
        w1_after = pred._weights[pred._active_buf[pp1]][pp1]['w_out']
        assert np.allclose(w1_before, w1_after)

    def test_training_with_momentum(self):
        meta = _small_meta()
        cfg_no_mom = _small_predictor_config(
            training_batch_size=4, momentum=0.0)
        cfg_mom = _small_predictor_config(
            training_batch_size=4, momentum=0.9)
        pred_no = ProbePredictor(meta, cfg_no_mom)
        pred_mom = ProbePredictor(meta, cfg_mom)
        pp = pred_no.probe_layer_indices[0]
        rng = np.random.default_rng(88)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in pred_no.target_layers_for(pp):
                pred_no.record_sample(pp, h, t, mask)
            for t in pred_mom.target_layers_for(pp):
                pred_mom.record_sample(pp, h, t, mask)
        for _ in range(3):
            pred_no.maybe_train(pp)
            pred_mom.maybe_train(pp)
        w_no = pred_no._weights[pred_no._active_buf[pp]][pp]['w_out']
        w_mom = pred_mom._weights[pred_mom._active_buf[pp]][pp]['w_out']
        assert not np.allclose(w_no, w_mom)

    def test_multiple_train_steps(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=4, min_samples_before_predict=1,
            learning_rate=0.05, momentum=0.0)
        pred = ProbePredictor(meta, cfg)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        target = targets[0]
        rng = np.random.default_rng(99)

        target_mask = np.zeros(16, dtype=np.float32)
        target_mask[0] = 1.0
        h_test = rng.standard_normal(64).astype(np.float32)
        for _ in range(30):
            h = h_test + rng.standard_normal(64).astype(np.float32) * 0.1
            for t in targets:
                pred.record_sample(pp, h, t, target_mask)

        scores = []
        for _ in range(10):
            pred.maybe_train(pp)
            out = pred.predict(pp, h_test)
            scores.append(out[target][0])
        assert scores[-1] > scores[0]


# ── Probe integration ─────────────────────────────────────────────────────


class TestProbeIntegration:
    def test_predict_delegates_to_predictor(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg_pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)

        p = Probe(meta, ProbeConfig(top_k=4, score_threshold=0.0),
                  predictor=pred)
        hints = p.predict(pp, h)
        assert len(hints) > 0
        assert all(h.confidence == PrefetchConfidence.MEDIUM for h in hints)
        assert all(h.source == PrefetchSource.PROBE for h in hints)

    def test_predict_without_predictor_returns_empty(self):
        p = Probe(_small_meta())
        h = np.random.randn(64).astype(np.float32)
        hints = p.predict(2, h)
        assert hints == []

    def test_hint_metadata(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg_pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[3] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)

        p = Probe(meta, ProbeConfig(top_k=2, score_threshold=0.0),
                  predictor=pred)
        hints = p.predict(pp, h)
        assert len(hints) > 0
        for hint in hints:
            assert hint.target_layer in targets
            assert hint.key.layer_idx == hint.target_layer
            assert 0 <= hint.key.expert_idx < 16
            assert hint.confidence == PrefetchConfidence.MEDIUM
            assert hint.source == PrefetchSource.PROBE

    def test_hints_sorted_by_score_descending(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg_pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)

        p = Probe(meta, ProbeConfig(top_k=16, score_threshold=0.0),
                  predictor=pred)
        hints = p.predict(pp, h)
        scores = [h.score for h in hints]
        assert scores == sorted(scores, reverse=True)

    def test_record_training_sample_delegates(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config()
        pred = ProbePredictor(meta, cfg_pred)
        p = Probe(meta, predictor=pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for t in targets:
            p.record_training_sample(pp, h, t, mask)
        assert pred.sample_count(pp) == 1

    def test_record_training_samples_batch(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config()
        pred = ProbePredictor(meta, cfg_pred)
        p = Probe(meta, predictor=pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        samples = [(t, mask) for t in targets]
        p.record_training_samples(pp, h, samples)
        assert pred.sample_count(pp) == 1

    def test_batch_matches_individual(self):
        meta = _small_meta()
        cfg_a = _small_predictor_config(min_samples_before_predict=1)
        cfg_b = _small_predictor_config(min_samples_before_predict=1)
        pred_a = ProbePredictor(meta, cfg_a)
        pred_b = ProbePredictor(meta, cfg_b)
        pp = pred_a.probe_layer_indices[0]
        targets = pred_a.target_layers_for(pp)
        rng = np.random.default_rng(55)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[2] = 1.0
        for t in targets:
            pred_a.record_sample(pp, h, t, mask)
        pred_b.record_samples_batch(pp, h, [(t, mask) for t in targets])
        for t in targets:
            buf_a = pred_a._buffers[(pp, t)]
            buf_b = pred_b._buffers[(pp, t)]
            assert np.allclose(buf_a.inputs[0], buf_b.inputs[0])
            assert np.allclose(buf_a.labels[0], buf_b.labels[0])


# ── Score-to-hint conversion ─────────────────────────────────────────────


class TestScoreToHints:
    def test_top_k_selection(self):
        p = Probe(_small_meta(), ProbeConfig(top_k=3, score_threshold=0.0))
        scores = np.zeros(16, dtype=np.float32)
        scores[0] = 0.9
        scores[1] = 0.8
        scores[2] = 0.7
        scores[3] = 0.6
        hints = p._scores_to_hints(5, scores)
        assert len(hints) == 3
        expert_ids = {h.key.expert_idx for h in hints}
        assert expert_ids == {0, 1, 2}

    def test_score_threshold_filters(self):
        p = Probe(_small_meta(), ProbeConfig(top_k=16, score_threshold=0.5))
        scores = np.zeros(16, dtype=np.float32)
        scores[0] = 0.9
        scores[1] = 0.6
        scores[2] = 0.4
        scores[3] = 0.1
        hints = p._scores_to_hints(5, scores)
        expert_ids = {h.key.expert_idx for h in hints}
        assert 0 in expert_ids
        assert 1 in expert_ids
        assert 2 not in expert_ids

    def test_all_zeros_returns_empty(self):
        p = Probe(_small_meta(), ProbeConfig(top_k=8, score_threshold=0.01))
        scores = np.zeros(16, dtype=np.float32)
        hints = p._scores_to_hints(5, scores)
        assert hints == []

    def test_multi_target_hints_present(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = ProbePredictor(meta, cfg_pred)
        pp = pred.probe_layer_indices[0]
        targets = pred.target_layers_for(pp)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(pp, h, t, mask)

        p = Probe(meta, ProbeConfig(top_k=2, score_threshold=0.0),
                  predictor=pred)
        hints = p.predict(pp, h)
        target_layers_in_hints = {h.target_layer for h in hints}
        assert len(target_layers_in_hints) > 1
