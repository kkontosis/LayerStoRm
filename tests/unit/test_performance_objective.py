"""Tests for orchestrator.performance_objective — weighted latency/throughput scoring."""

import pytest

from orchestrator.performance_objective import ObjectiveConfig, PerformanceObjective


class TestDefaults:
    def test_default_config(self):
        obj = PerformanceObjective()
        assert obj.latency_weight == pytest.approx(0.6)
        assert obj.throughput_weight == pytest.approx(0.4)

    def test_default_is_latency_dominant(self):
        obj = PerformanceObjective()
        assert obj.is_latency_dominant is True
        assert obj.is_throughput_dominant is False


class TestScoring:
    def test_latency_only(self):
        obj = PerformanceObjective(ObjectiveConfig(1.0, 0.0))
        assert obj.score(0.8, 0.2) == pytest.approx(0.8)
        assert obj.score(0.0, 1.0) == pytest.approx(0.0)

    def test_throughput_only(self):
        obj = PerformanceObjective(ObjectiveConfig(0.0, 1.0))
        assert obj.score(0.8, 0.2) == pytest.approx(0.2)
        assert obj.score(1.0, 0.0) == pytest.approx(0.0)

    def test_equal_weights(self):
        obj = PerformanceObjective(ObjectiveConfig(0.5, 0.5))
        assert obj.score(1.0, 0.0) == pytest.approx(0.5)
        assert obj.score(0.0, 1.0) == pytest.approx(0.5)
        assert obj.score(0.6, 0.4) == pytest.approx(0.5)

    def test_weighted_sum(self):
        obj = PerformanceObjective(ObjectiveConfig(0.6, 0.4))
        assert obj.score(1.0, 1.0) == pytest.approx(1.0)
        assert obj.score(0.0, 0.0) == pytest.approx(0.0)
        assert obj.score(1.0, 0.0) == pytest.approx(0.6)
        assert obj.score(0.0, 1.0) == pytest.approx(0.4)

    def test_latency_dominant_prefers_latency(self):
        obj = PerformanceObjective(ObjectiveConfig(0.8, 0.2))
        assert obj.score(1.0, 0.0) > obj.score(0.0, 1.0)

    def test_throughput_dominant_prefers_throughput(self):
        obj = PerformanceObjective(ObjectiveConfig(0.2, 0.8))
        assert obj.score(0.0, 1.0) > obj.score(1.0, 0.0)


class TestNormalization:
    def test_unnormalized_weights(self):
        obj = PerformanceObjective(ObjectiveConfig(3.0, 7.0))
        assert obj.latency_weight == pytest.approx(0.3)
        assert obj.throughput_weight == pytest.approx(0.7)
        assert obj.latency_weight + obj.throughput_weight == pytest.approx(1.0)

    def test_both_zero_equal_split(self):
        obj = PerformanceObjective(ObjectiveConfig(0.0, 0.0))
        assert obj.latency_weight == pytest.approx(0.5)
        assert obj.throughput_weight == pytest.approx(0.5)
        assert obj.is_latency_dominant is False
        assert obj.is_throughput_dominant is False


class TestDominance:
    def test_latency_dominant(self):
        obj = PerformanceObjective(ObjectiveConfig(0.7, 0.3))
        assert obj.is_latency_dominant is True
        assert obj.is_throughput_dominant is False

    def test_throughput_dominant(self):
        obj = PerformanceObjective(ObjectiveConfig(0.3, 0.7))
        assert obj.is_latency_dominant is False
        assert obj.is_throughput_dominant is True

    def test_equal_neither_dominant(self):
        obj = PerformanceObjective(ObjectiveConfig(0.5, 0.5))
        assert obj.is_latency_dominant is False
        assert obj.is_throughput_dominant is False
