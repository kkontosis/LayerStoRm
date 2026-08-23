"""Tests for orchestrator.prescope — exact gating lookahead + LLaPor predictor."""

import numpy as np
import pytest

from orchestrator.types import (
    ExpertKey,
    PrefetchConfidence,
    PrefetchSource,
)
from orchestrator.prescope import (
    LayerGroup,
    PreScope,
    PrescopeConfig,
    PrescopePredictor,
    PredictorConfig,
)


def _meta(num_layers: int = 61, num_moe_layers: int = 58,
          num_experts: int = 256):
    from orchestrator.types import EngineMetadata, GpuConfig
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


class TestTargetLayer:
    def test_moe_layer_returns_next(self):
        ps = PreScope(_meta())
        assert ps.target_layer(10) == 11

    def test_dense_layer_returns_none(self):
        ps = PreScope(_meta())
        assert ps.target_layer(0) is None
        assert ps.target_layer(1) is None

    def test_last_layer_returns_none(self):
        ps = PreScope(_meta())
        assert ps.target_layer(60) is None

    def test_boundary_first_moe(self):
        ps = PreScope(_meta())
        assert ps.target_layer(2) == 3

    def test_boundary_last_moe(self):
        ps = PreScope(_meta())
        assert ps.target_layer(59) == 60

    def test_is_moe_layer(self):
        ps = PreScope(_meta())
        assert not ps.is_moe_layer(0)
        assert not ps.is_moe_layer(2)
        assert ps.is_moe_layer(3)
        assert ps.is_moe_layer(30)
        assert ps.is_moe_layer(60)
        assert not ps.is_moe_layer(61)

    def test_first_moe_layer_property(self):
        ps = PreScope(_meta())
        assert ps.first_moe_layer == 3


class TestGatingProcessing:
    def test_top_k_selection(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=4, score_threshold=0.0))
        scores = np.zeros(256, dtype=np.float32)
        scores[10] = 0.9
        scores[20] = 0.8
        scores[30] = 0.7
        scores[40] = 0.6
        scores[50] = 0.5
        hints = ps.process_gating_output(5, scores)
        assert len(hints) == 4
        expert_ids = {h.key.expert_idx for h in hints}
        assert expert_ids == {10, 20, 30, 40}

    def test_score_threshold_filters_low(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=8, score_threshold=0.5))
        scores = np.zeros(256, dtype=np.float32)
        scores[0] = 0.9
        scores[1] = 0.6
        scores[2] = 0.4
        scores[3] = 0.1
        hints = ps.process_gating_output(5, scores)
        expert_ids = {h.key.expert_idx for h in hints}
        assert 0 in expert_ids
        assert 1 in expert_ids
        assert 2 not in expert_ids
        assert 3 not in expert_ids

    def test_scores_descending_order(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=8, score_threshold=0.0))
        scores = np.zeros(256, dtype=np.float32)
        scores[5] = 0.3
        scores[10] = 0.9
        scores[15] = 0.6
        hints = ps.process_gating_output(5, scores)
        assert hints[0].score == pytest.approx(0.9)
        assert hints[1].score == pytest.approx(0.6)
        assert hints[2].score == pytest.approx(0.3)

    def test_hint_metadata(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=2, score_threshold=0.0))
        scores = np.zeros(256, dtype=np.float32)
        scores[7] = 0.8
        hints = ps.process_gating_output(10, scores)
        h = hints[0]
        assert h.key == ExpertKey(10, 7)
        assert h.target_layer == 10
        assert h.confidence == PrefetchConfidence.HIGH
        assert h.source == PrefetchSource.PRESCOPE
        assert h.score == pytest.approx(0.8)

    def test_single_token_1d(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=2, score_threshold=0.0))
        scores = np.array([0.1] * 256, dtype=np.float32)
        scores[0] = 0.9
        scores[1] = 0.8
        hints = ps.process_gating_output(5, scores)
        assert len(hints) == 2

    def test_batch_aggregation(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=2, score_threshold=0.0))
        scores = np.zeros((3, 256), dtype=np.float32)
        scores[0, 10] = 0.5
        scores[1, 10] = 0.9
        scores[2, 20] = 0.7
        hints = ps.process_gating_output(5, scores)
        by_expert = {h.key.expert_idx: h.score for h in hints}
        assert by_expert[10] == pytest.approx(0.9)
        assert by_expert[20] == pytest.approx(0.7)

    def test_all_zeros_returns_empty(self):
        ps = PreScope(_meta(), PrescopeConfig(top_k=8, score_threshold=0.01))
        scores = np.zeros(256, dtype=np.float32)
        hints = ps.process_gating_output(5, scores)
        assert hints == []


class TestConfig:
    def test_default_config(self):
        ps = PreScope(_meta())
        assert ps.config.enabled is True
        assert ps.config.top_k == 8
        assert ps.config.score_threshold == pytest.approx(0.01)

    def test_custom_config(self):
        cfg = PrescopeConfig(enabled=True, top_k=4, score_threshold=0.05)
        ps = PreScope(_meta(), cfg)
        assert ps.config.top_k == 4
        assert ps.config.score_threshold == pytest.approx(0.05)

    def test_disabled_prescope(self):
        ps = PreScope(_meta(), PrescopeConfig(enabled=False))
        assert not ps.enabled


def _small_meta():
    """Small model for predictor tests (fewer experts = faster)."""
    from orchestrator.types import EngineMetadata, GpuConfig
    return EngineMetadata(
        num_gpus=1, num_moe_layers=8, num_experts=16,
        num_layers=10, expert_bytes=1000, kv_bytes_per_page=100,
        gpus=(GpuConfig(position=0, gpu_type="test", is_tp=False,
                        vram_bytes=1024**3),),
    )


def _small_predictor_config(**overrides):
    defaults = dict(
        hidden_size=64, pca_dim=16, hidden_dim=16,
        learning_rate=0.01, momentum=0.0,
        training_batch_size=8, training_buffer_size=128,
        focal_loss_gamma=2.0, balance_loss_lambda=0.5,
        input_group_frac=0.25, output_group_frac=0.25,
        min_samples_before_predict=8, freq_ema_alpha=0.01,
    )
    defaults.update(overrides)
    return PredictorConfig(**defaults)


class TestPredictor:
    def test_predict_cold_start_returns_zeros(self):
        pred = PrescopePredictor(_small_meta(), _small_predictor_config())
        h = np.random.randn(64).astype(np.float32)
        out = pred.predict(2, h)
        assert out.shape == (16,)
        assert np.all(out == 0.0)

    def test_predict_output_shape(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        pred.record_sample(2, h, mask)
        out = pred.predict(2, h)
        assert out.shape == (16,)

    def test_predict_output_range(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        rng = np.random.default_rng(99)
        for _ in range(5):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            pred.record_sample(2, h, mask)
        out = pred.predict(2, rng.standard_normal(64).astype(np.float32))
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_record_sample_increments_count(self):
        pred = PrescopePredictor(_small_meta(), _small_predictor_config())
        assert pred.sample_count(2) == 0
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        pred.record_sample(2, h, mask)
        assert pred.sample_count(2) == 1
        pred.record_sample(2, h, mask)
        assert pred.sample_count(2) == 2

    def test_ring_buffer_wraps(self):
        cfg = _small_predictor_config(training_buffer_size=4)
        pred = PrescopePredictor(_small_meta(), cfg)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        for _ in range(10):
            pred.record_sample(2, h, mask)
        assert pred.sample_count(2) == 4

    def test_maybe_train_updates_weights(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = PrescopePredictor(meta, cfg)
        rng = np.random.default_rng(123)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            pred.record_sample(3, h, mask)
        w_before = pred._weights[pred._active_buf][3]['w1'].copy()
        trained = pred.maybe_train(3)
        assert trained
        w_after = pred._weights[pred._active_buf][3]['w1']
        assert not np.allclose(w_before, w_after)

    def test_double_buffer_swap(self):
        meta = _small_meta()
        cfg = _small_predictor_config(training_batch_size=4)
        pred = PrescopePredictor(meta, cfg)
        rng = np.random.default_rng(456)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = (rng.random(16) > 0.5).astype(np.float32)
            pred.record_sample(3, h, mask)
        buf_before = pred._active_buf
        pred.maybe_train(3)
        assert pred._active_buf != buf_before

    def test_layer_group_assignment(self):
        meta = _small_meta()  # 8 MoE layers: 2-9, first_moe=2
        cfg = _small_predictor_config(input_group_frac=0.25,
                                      output_group_frac=0.25)
        pred = PrescopePredictor(meta, cfg)
        assert pred.layer_group(2) == LayerGroup.INPUT
        assert pred.layer_group(3) == LayerGroup.INPUT
        assert pred.layer_group(4) == LayerGroup.MIDDLE
        assert pred.layer_group(7) == LayerGroup.MIDDLE
        assert pred.layer_group(8) == LayerGroup.OUTPUT
        assert pred.layer_group(9) == LayerGroup.OUTPUT

    def test_focal_loss_harder_samples_weighted(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=2, focal_loss_gamma=2.0,
            balance_loss_lambda=0.0)
        pred = PrescopePredictor(meta, cfg)
        rng = np.random.default_rng(789)
        for _ in range(10):
            h = rng.standard_normal(64).astype(np.float32)
            mask = np.zeros(16, dtype=np.float32)
            mask[0] = 1.0
            pred.record_sample(3, h, mask)

        w = pred._weights[pred._active_buf][3]
        X = pred._buffers[3].inputs[:2]
        Y = pred._buffers[3].labels[:2]
        grads = pred._train_step(w, X, Y, pred.layer_group(3), 3)
        assert 'w2' in grads
        assert not np.allclose(grads['w2'], 0.0)

    def test_predict_after_training_improves(self):
        meta = _small_meta()
        cfg = _small_predictor_config(
            training_batch_size=8, min_samples_before_predict=1,
            learning_rate=0.05, momentum=0.0)
        pred = PrescopePredictor(meta, cfg)
        rng = np.random.default_rng(42)
        target_mask = np.zeros(16, dtype=np.float32)
        target_mask[0] = 1.0
        target_mask[1] = 1.0

        h_test = rng.standard_normal(64).astype(np.float32)
        for _ in range(50):
            h = h_test + rng.standard_normal(64).astype(np.float32) * 0.1
            pred.record_sample(3, h, target_mask)

        pred_before = pred.predict(3, h_test).copy()
        for _ in range(30):
            pred.maybe_train(3)
        pred_after = pred.predict(3, h_test)
        score_before = pred_before[0] + pred_before[1]
        score_after = pred_after[0] + pred_after[1]
        assert score_after > score_before

    def test_middle_layer_residual_block(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        mid_layer = 5
        assert pred.layer_group(mid_layer) == LayerGroup.MIDDLE
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        pred.record_sample(mid_layer, h, mask)
        out = pred.predict(mid_layer, h)
        assert out.shape == (16,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_prev_routing_weights_used(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        rw = np.zeros(16, dtype=np.float32)
        rw[3] = 0.8
        pred.record_sample(3, h, mask, rw)
        out_with = pred.predict(3, h, rw)
        out_without = pred.predict(3, h)
        assert not np.allclose(out_with, out_without)


class TestPredictorIntegration:
    def test_predict_from_hidden_state_delegates(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        mask[0] = 1.0
        pred.record_sample(3, h, mask)

        ps = PreScope(meta, PrescopeConfig(top_k=4, score_threshold=0.0),
                      predictor=pred)
        hints = ps.predict_from_hidden_state(3, h)
        assert len(hints) > 0
        assert all(h.confidence == PrefetchConfidence.MEDIUM for h in hints)
        assert all(h.source == PrefetchSource.PRESCOPE for h in hints)

    def test_predict_from_hidden_state_without_predictor(self):
        ps = PreScope(_small_meta())
        h = np.random.randn(64).astype(np.float32)
        hints = ps.predict_from_hidden_state(3, h)
        assert hints == []

    def test_record_training_sample_delegates(self):
        meta = _small_meta()
        cfg = _small_predictor_config()
        pred = PrescopePredictor(meta, cfg)
        ps = PreScope(meta, predictor=pred)
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        ps.record_training_sample(3, h, mask)
        assert pred.sample_count(3) == 1

    def test_record_without_predictor_noop(self):
        ps = PreScope(_small_meta())
        h = np.random.randn(64).astype(np.float32)
        mask = np.zeros(16, dtype=np.float32)
        ps.record_training_sample(3, h, mask)

    def test_predict_non_moe_layer_returns_empty(self):
        meta = _small_meta()
        cfg = _small_predictor_config(min_samples_before_predict=1)
        pred = PrescopePredictor(meta, cfg)
        ps = PreScope(meta, predictor=pred)
        h = np.random.randn(64).astype(np.float32)
        hints = ps.predict_from_hidden_state(0, h)
        assert hints == []
