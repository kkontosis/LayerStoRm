"""Tests for orchestrator.residual_correction — online-trained MLP for draft."""

import numpy as np
import pytest

from orchestrator.residual_correction import (
    ResidualCorrection,
    ResidualCorrectionConfig,
)

D_INT = 64
N_EXP = 16
N_MOE = 4


def _rc(**kwargs) -> ResidualCorrection:
    cfg = ResidualCorrectionConfig(**kwargs)
    return ResidualCorrection(cfg, num_moe_layers=N_MOE,
                              output_dim=D_INT, num_experts=N_EXP)


def _fill(rc: ResidualCorrection, layer: int, n: int,
          residual_scale: float = 1.0) -> None:
    rng = np.random.default_rng(123 + layer)
    for _ in range(n):
        top1 = rng.standard_normal(D_INT).astype(np.float32)
        gating = rng.random(N_EXP).astype(np.float32)
        residual = rng.standard_normal(D_INT).astype(np.float32) * residual_scale
        rc.record_sample(layer, top1, gating, top1 + residual)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredict:

    def test_cold_start_returns_zeros(self):
        rc = _rc()
        top1 = np.ones(D_INT, dtype=np.float32)
        gating = np.ones(N_EXP, dtype=np.float32)
        result = rc.predict(0, top1, gating)
        assert result.shape == (D_INT,)
        np.testing.assert_array_equal(result, 0.0)

    def test_warm_returns_nonzero(self):
        rc = _rc(min_samples_before_predict=10)
        _fill(rc, 0, 20, residual_scale=5.0)
        for _ in range(3):
            rc.maybe_train(0)
        top1 = np.ones(D_INT, dtype=np.float32)
        gating = np.ones(N_EXP, dtype=np.float32) * 0.5
        result = rc.predict(0, top1, gating)
        assert result.shape == (D_INT,)
        assert not np.allclose(result, 0.0)

    def test_output_shape(self):
        rc = _rc(min_samples_before_predict=5)
        _fill(rc, 0, 10)
        rc.maybe_train(0)
        top1 = np.zeros(D_INT, dtype=np.float32)
        gating = np.zeros(N_EXP, dtype=np.float32)
        assert rc.predict(0, top1, gating).shape == (D_INT,)

    def test_layers_independent(self):
        rc = _rc(min_samples_before_predict=5)
        _fill(rc, 0, 10, residual_scale=10.0)
        rc.maybe_train(0)
        top1 = np.ones(D_INT, dtype=np.float32)
        gating = np.ones(N_EXP, dtype=np.float32)
        pred0 = rc.predict(0, top1, gating)
        pred1 = rc.predict(1, top1, gating)
        assert not np.allclose(pred0, 0.0)
        np.testing.assert_array_equal(pred1, 0.0)

    def test_disabled_returns_zeros(self):
        rc = _rc(enabled=False, min_samples_before_predict=5)
        _fill(rc, 0, 10)
        rc.maybe_train(0)
        top1 = np.ones(D_INT, dtype=np.float32)
        gating = np.ones(N_EXP, dtype=np.float32)
        np.testing.assert_array_equal(rc.predict(0, top1, gating), 0.0)


# ---------------------------------------------------------------------------
# Record sample
# ---------------------------------------------------------------------------


class TestRecordSample:

    def test_sample_count_increments(self):
        rc = _rc()
        assert rc.sample_count(0) == 0
        _fill(rc, 0, 5)
        assert rc.sample_count(0) == 5

    def test_layers_have_independent_buffers(self):
        rc = _rc()
        _fill(rc, 0, 3)
        _fill(rc, 1, 7)
        assert rc.sample_count(0) == 3
        assert rc.sample_count(1) == 7

    def test_ring_buffer_wraps(self):
        rc = _rc(training_buffer_size=5000)
        _fill(rc, 0, 6000)
        assert rc.sample_count(0) == 5000


# ---------------------------------------------------------------------------
# Maybe train
# ---------------------------------------------------------------------------


class TestMaybeTrain:

    def test_returns_false_insufficient_samples(self):
        rc = _rc(training_batch_size=32)
        _fill(rc, 0, 10)
        assert rc.maybe_train(0) is False

    def test_returns_true_sufficient_samples(self):
        rc = _rc(training_batch_size=16)
        _fill(rc, 0, 20)
        assert rc.maybe_train(0) is True

    def test_double_buffer_swap(self):
        rc = _rc(training_batch_size=16, min_samples_before_predict=5)
        _fill(rc, 0, 20)
        before = rc._active_buf[0]
        rc.maybe_train(0)
        assert rc._active_buf[0] == 1 - before

    def test_predictions_change_after_training(self):
        rc = _rc(training_batch_size=16, min_samples_before_predict=5,
                 learning_rate=0.01)
        _fill(rc, 0, 100, residual_scale=10.0)
        rc.maybe_train(0)
        top1 = np.ones(D_INT, dtype=np.float32)
        gating = np.ones(N_EXP, dtype=np.float32)
        pred1 = rc.predict(0, top1, gating).copy()
        for _ in range(10):
            rc.maybe_train(0)
        pred2 = rc.predict(0, top1, gating)
        assert not np.array_equal(pred1, pred2)

    def test_training_only_affects_target_layer(self):
        rc = _rc(training_batch_size=16, min_samples_before_predict=5)
        _fill(rc, 0, 20)
        _fill(rc, 1, 20)
        w1_before = rc._weights[rc._active_buf[1]][1]['w1'].copy()
        rc.maybe_train(0)
        w1_after = rc._weights[rc._active_buf[1]][1]['w1']
        np.testing.assert_array_equal(w1_before, w1_after)


# ---------------------------------------------------------------------------
# Double buffer
# ---------------------------------------------------------------------------


class TestDoubleBuffer:

    def test_initial_active_buffer_zero(self):
        rc = _rc()
        for layer in range(N_MOE):
            assert rc._active_buf[layer] == 0

    def test_swap_toggles(self):
        rc = _rc(training_batch_size=8, min_samples_before_predict=5)
        _fill(rc, 0, 20)
        rc.maybe_train(0)
        assert rc._active_buf[0] == 1
        rc.maybe_train(0)
        assert rc._active_buf[0] == 0

    def test_trained_weights_differ_from_initial(self):
        rc = _rc(training_batch_size=8, min_samples_before_predict=5,
                 learning_rate=0.01)
        initial_w1 = rc._weights[0][0]['w1'].copy()
        _fill(rc, 0, 50, residual_scale=10.0)
        for _ in range(5):
            rc.maybe_train(0)
        active = rc._active_buf[0]
        trained_w1 = rc._weights[active][0]['w1']
        assert not np.array_equal(initial_w1, trained_w1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = ResidualCorrectionConfig()
        assert cfg.enabled is True
        assert cfg.hidden_size == 128
        assert cfg.learning_rate == pytest.approx(5e-5)
        assert cfg.training_buffer_size == 500
        assert cfg.training_batch_size == 32
        assert cfg.min_samples_before_predict == 64

    def test_frozen_config(self):
        cfg = ResidualCorrectionConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_is_enabled_property(self):
        assert _rc(enabled=True).is_enabled is True
        assert _rc(enabled=False).is_enabled is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_full_cycle_prediction_improves(self):
        rc = _rc(training_batch_size=16, min_samples_before_predict=10,
                 learning_rate=0.01, hidden_size=64)
        rng = np.random.default_rng(99)
        residual_direction = rng.standard_normal(D_INT).astype(np.float32)
        residual_direction /= np.linalg.norm(residual_direction)

        for _ in range(200):
            top1 = rng.standard_normal(D_INT).astype(np.float32)
            gating = rng.random(N_EXP).astype(np.float32)
            residual = residual_direction * 5.0
            rc.record_sample(0, top1, gating, top1 + residual)

        for _ in range(20):
            rc.maybe_train(0)

        top1 = rng.standard_normal(D_INT).astype(np.float32)
        gating = rng.random(N_EXP).astype(np.float32)
        pred = rc.predict(0, top1, gating)
        zero_err = np.linalg.norm(residual_direction * 5.0)
        pred_err = np.linalg.norm(pred - residual_direction * 5.0)
        assert pred_err < zero_err

    def test_zero_residual_predicts_near_zero(self):
        rc = _rc(training_batch_size=16, min_samples_before_predict=10,
                 learning_rate=0.01, hidden_size=64)
        rng = np.random.default_rng(42)
        for _ in range(100):
            top1 = rng.standard_normal(D_INT).astype(np.float32)
            gating = rng.random(N_EXP).astype(np.float32)
            rc.record_sample(0, top1, gating, top1)

        for _ in range(20):
            rc.maybe_train(0)

        top1 = rng.standard_normal(D_INT).astype(np.float32)
        gating = rng.random(N_EXP).astype(np.float32)
        pred = rc.predict(0, top1, gating)
        assert np.linalg.norm(pred) < 1.0

    def test_train_multiple_layers(self):
        rc = _rc(training_batch_size=8, min_samples_before_predict=5)
        for layer in range(N_MOE):
            _fill(rc, layer, 20)
            assert rc.maybe_train(layer) is True
        assert rc.num_layers_trained == N_MOE

    def test_num_layers_trained_property(self):
        rc = _rc(training_batch_size=8, min_samples_before_predict=5)
        assert rc.num_layers_trained == 0
        _fill(rc, 0, 20)
        rc.maybe_train(0)
        assert rc.num_layers_trained == 1
        _fill(rc, 2, 20)
        rc.maybe_train(2)
        assert rc.num_layers_trained == 2

    def test_sample_count_across_layers(self):
        rc = _rc()
        for layer in range(N_MOE):
            _fill(rc, layer, (layer + 1) * 10)
        for layer in range(N_MOE):
            assert rc.sample_count(layer) == (layer + 1) * 10
