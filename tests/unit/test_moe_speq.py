"""Tests for orchestrator.moe_speq — per-pair online-trained expert predictor."""

import numpy as np
import pytest

from orchestrator.types import (
    EngineMetadata,
    ExpertKey,
    GpuConfig,
    PrefetchConfidence,
    PrefetchSource,
)
from orchestrator.moe_speq import (
    MoeSpeq,
    MoeSpeqConfig,
    MoeSpeqPredictor,
    MoeSpeqPredictorConfig,
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
        learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
        training_batch_size=8, training_buffer_size=128,
        min_samples_before_predict=8,
    )
    defaults.update(overrides)
    return MoeSpeqPredictorConfig(**defaults)


# ── Source/target layer identification ────────────────────────────────────


class TestSourceTargetLayers:
    def test_source_layers_are_all_moe(self):
        meta = _small_meta()
        pred = MoeSpeqPredictor(meta, _small_predictor_config())
        first_moe = meta.num_layers - meta.num_moe_layers
        expected = tuple(range(first_moe, meta.num_layers))
        assert pred.source_layers == expected

    def test_target_layers_strictly_after_source(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        for src in pred.source_layers:
            for tgt in pred.target_layers_for(src):
                assert tgt > src

    def test_target_layers_are_moe(self):
        meta = _small_meta()
        first_moe = meta.num_layers - meta.num_moe_layers
        pred = MoeSpeqPredictor(meta, _small_predictor_config())
        for src in pred.source_layers:
            for tgt in pred.target_layers_for(src):
                assert first_moe <= tgt < meta.num_layers

    def test_first_moe_has_max_targets(self):
        meta = _small_meta()
        pred = MoeSpeqPredictor(meta, _small_predictor_config())
        first_src = pred.source_layers[0]
        assert len(pred.target_layers_for(first_src)) == meta.num_moe_layers - 1

    def test_last_moe_has_no_targets(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        last_src = pred.source_layers[-1]
        assert pred.target_layers_for(last_src) == []


# ── Config ────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        s = MoeSpeq(_meta())
        assert s.config.enabled is True
        assert s.config.top_k == 8
        assert s.config.score_threshold == pytest.approx(0.01)
        assert s.config.high_confidence_threshold == pytest.approx(0.8)
        assert s.config.medium_confidence_threshold == pytest.approx(0.5)

    def test_custom_config(self):
        cfg = MoeSpeqConfig(top_k=4, score_threshold=0.05,
                            high_confidence_threshold=0.9)
        s = MoeSpeq(_meta(), cfg)
        assert s.config.top_k == 4

    def test_disabled(self):
        s = MoeSpeq(_meta(), MoeSpeqConfig(enabled=False))
        assert not s.enabled


# ── Predictor core ────────────────────────────────────────────────────────


class TestPredictor:
    def test_predict_cold_start_returns_empty(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        src = pred.source_layers[0]
        h = np.random.randn(64).astype(np.float32)
        assert pred.predict(src, h) == {}

    def test_predict_output_shape(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(99)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(src, h, t, mask)
        out = pred.predict(src, h)
        assert set(out.keys()) == set(targets)
        for scores in out.values():
            assert scores.shape == (16,)

    def test_predict_output_range(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(99)
        for _ in range(3):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(src, h, t, mask)
        out = pred.predict(src, rng.standard_normal(64).astype(np.float32))
        for scores in out.values():
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

    def test_record_sample_increments_count(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        assert pred.sample_count(src) == 0
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for t in targets:
            pred.record_sample(src, h, t, mask)
        assert pred.sample_count(src) == 1

    def test_ring_buffer_wraps(self):
        cfg = _small_predictor_config(training_buffer_size=4)
        pred = MoeSpeqPredictor(_small_meta(), cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for _ in range(10):
            for t in targets:
                pred.record_sample(src, h, t, mask)
        assert pred.sample_count(src) == 4

    def test_maybe_train_updates_weights(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(123)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(src, h, t, mask)
        w_before = pred._weights[pred._active_buf[src]][src]['w_out'].copy()
        trained = pred.maybe_train(src)
        assert trained
        w_after = pred._weights[pred._active_buf[src]][src]['w_out']
        assert not np.allclose(w_before, w_after)

    def test_double_buffer_swap(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(456)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(src, h, t, mask)
        buf_before = pred._active_buf[src]
        pred.maybe_train(src)
        assert pred._active_buf[src] != buf_before

    def test_maybe_train_returns_false_insufficient_data(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        src = pred.source_layers[0]
        assert not pred.maybe_train(src)

    def test_training_independent_per_source(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = MoeSpeqPredictor(meta, cfg)
        src0 = pred.source_layers[0]
        src1 = pred.source_layers[1]
        rng = np.random.default_rng(77)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in pred.target_layers_for(src0):
                pred.record_sample(src0, h, t, mask)
        w1_before = pred._weights[pred._active_buf[src1]][src1]['w_out'].copy()
        pred.maybe_train(src0)
        w1_after = pred._weights[pred._active_buf[src1]][src1]['w_out']
        assert np.allclose(w1_before, w1_after)

    def test_relu_no_negative_features(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(src, h, t, mask)
        feat = pred._extract_features(src, h)
        assert np.all(feat >= 0.0)


# ── Predictor training ────────────────────────────────────────────────────


class TestPredictorTraining:
    def test_predict_after_training_improves(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=8, min_samples_before_predict=1,
            learning_rate=0.05)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        target = targets[0]
        rng = np.random.default_rng(42)

        target_mask = np.zeros(16, dtype=np.float32)
        target_mask[0] = 1.0
        target_mask[1] = 1.0

        h_test = rng.standard_normal(64).astype(np.float32)
        for _ in range(50):
            h = h_test + rng.standard_normal(64).astype(np.float32) * 0.1
            for t in targets:
                pred.record_sample(src, h, t, target_mask)

        pred_before = pred.predict(src, h_test)
        score_before = pred_before[target][0] + pred_before[target][1]
        for _ in range(30):
            pred.maybe_train(src)
        pred_after = pred.predict(src, h_test)
        score_after = pred_after[target][0] + pred_after[target][1]
        assert score_after > score_before

    def test_multiple_train_steps(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=4, min_samples_before_predict=1,
            learning_rate=0.05)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        target = targets[0]
        rng = np.random.default_rng(99)

        target_mask = np.zeros(16, dtype=np.float32)
        target_mask[0] = 1.0
        h_test = rng.standard_normal(64).astype(np.float32)
        for _ in range(30):
            h = h_test + rng.standard_normal(64).astype(np.float32) * 0.1
            for t in targets:
                pred.record_sample(src, h, t, target_mask)

        scores = []
        for _ in range(10):
            pred.maybe_train(src)
            out = pred.predict(src, h_test)
            scores.append(out[target][0])
        assert scores[-1] > scores[0]

    def test_adam_moments_nonzero_after_training(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = MoeSpeqPredictor(meta, cfg)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(55)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            for t in targets:
                pred.record_sample(src, h, t, mask)
        pred.maybe_train(src)
        assert pred._adam_step[src] == 1
        assert not np.allclose(pred._adam_m[src]['w_out'], 0.0)
        assert not np.allclose(pred._adam_v[src]['w_out'], 0.0)

    def test_last_source_no_targets_returns_false(self):
        pred = MoeSpeqPredictor(_small_meta(), _small_predictor_config())
        last_src = pred.source_layers[-1]
        assert not pred.maybe_train(last_src)


# ── Variable confidence ───────────────────────────────────────────────────


class TestVariableConfidence:
    def _hint_for_expert(self, hints, expert_idx):
        return next(h for h in hints if h.key.expert_idx == expert_idx)

    def test_high_confidence(self):
        s = MoeSpeq(_small_meta(), MoeSpeqConfig(
            high_confidence_threshold=0.8, top_k=16, score_threshold=0.0))
        scores = np.zeros(16, dtype=np.float32)
        scores[0] = 0.95
        hints = s._scores_to_hints(5, scores)
        assert self._hint_for_expert(hints, 0).confidence == PrefetchConfidence.HIGH

    def test_medium_confidence(self):
        s = MoeSpeq(_small_meta(), MoeSpeqConfig(
            high_confidence_threshold=0.8, medium_confidence_threshold=0.5,
            top_k=16, score_threshold=0.0))
        scores = np.zeros(16, dtype=np.float32)
        scores[0] = 0.65
        hints = s._scores_to_hints(5, scores)
        assert self._hint_for_expert(hints, 0).confidence == PrefetchConfidence.MEDIUM

    def test_low_confidence(self):
        s = MoeSpeq(_small_meta(), MoeSpeqConfig(
            high_confidence_threshold=0.8, medium_confidence_threshold=0.5,
            top_k=16, score_threshold=0.0))
        scores = np.zeros(16, dtype=np.float32)
        scores[0] = 0.3
        hints = s._scores_to_hints(5, scores)
        assert self._hint_for_expert(hints, 0).confidence == PrefetchConfidence.LOW


# ── Integration ───────────────────────────────────────────────────────────


class TestIntegration:
    def test_predict_delegates_to_predictor(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg_pred)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(src, h, t, mask)

        s = MoeSpeq(meta, MoeSpeqConfig(top_k=4, score_threshold=0.0),
                    predictor=pred)
        hints = s.predict(src, h)
        assert len(hints) > 0
        assert all(h.source == PrefetchSource.MOE_SPEQ for h in hints)

    def test_predict_without_predictor_returns_empty(self):
        s = MoeSpeq(_small_meta())
        h = np.random.randn(64).astype(np.float32)
        assert s.predict(2, h) == []

    def test_hint_metadata(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg_pred)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[3] = 1.0
        for t in targets:
            pred.record_sample(src, h, t, mask)

        s = MoeSpeq(meta, MoeSpeqConfig(top_k=2, score_threshold=0.0),
                    predictor=pred)
        hints = s.predict(src, h)
        assert len(hints) > 0
        for hint in hints:
            assert hint.target_layer in targets
            assert hint.key.layer_idx == hint.target_layer
            assert 0 <= hint.key.expert_idx < 16
            assert hint.source == PrefetchSource.MOE_SPEQ

    def test_hints_sorted_by_score_descending(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config(min_samples_before_predict=1)
        pred = MoeSpeqPredictor(meta, cfg_pred)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        rng = np.random.default_rng(42)
        h = rng.standard_normal(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        for t in targets:
            pred.record_sample(src, h, t, mask)

        s = MoeSpeq(meta, MoeSpeqConfig(top_k=16, score_threshold=0.0),
                    predictor=pred)
        hints = s.predict(src, h)
        scores = [h.score for h in hints]
        assert scores == sorted(scores, reverse=True)

    def test_record_training_samples_batch(self):
        meta = _small_meta()
        cfg_pred = _small_predictor_config()
        pred = MoeSpeqPredictor(meta, cfg_pred)
        s = MoeSpeq(meta, predictor=pred)
        src = pred.source_layers[0]
        targets = pred.target_layers_for(src)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        samples = [(t, mask) for t in targets]
        s.record_training_samples(src, h, samples)
        assert pred.sample_count(src) == 1

    def test_non_moe_layer_returns_empty(self):
        s = MoeSpeq(_small_meta())
        h = np.random.randn(64).astype(np.float32)
        assert s.predict(0, h) == []
