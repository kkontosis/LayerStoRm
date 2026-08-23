"""Tests for orchestrator.layer_skip — cosine similarity-based layer skipping."""

import numpy as np
import pytest

from orchestrator.layer_skip import LayerSkip, LayerSkipConfig

N_LAYERS = 20
NO_FIRST = 3
NO_LAST = 3


def _ls(**kwargs) -> LayerSkip:
    cfg = LayerSkipConfig(no_skip_first=NO_FIRST, no_skip_last=NO_LAST,
                          **kwargs)
    return LayerSkip(cfg, num_layers=N_LAYERS)


def _high_sims(*layers: int, value: float = 0.999) -> dict[int, float]:
    return {l: value for l in layers}


# ---------------------------------------------------------------------------
# Compute skip set
# ---------------------------------------------------------------------------


class TestComputeSkipSet:

    def test_high_similarity_skips_next(self):
        ls = _ls(threshold=0.995)
        sims = _high_sims(5)
        skip = ls.compute_skip_set(sims)
        assert 6 in skip

    def test_low_similarity_no_skip(self):
        ls = _ls(threshold=0.995)
        sims = {5: 0.990}
        skip = ls.compute_skip_set(sims)
        assert 6 not in skip

    def test_at_threshold_skips(self):
        ls = _ls(threshold=0.995)
        sims = {5: 0.995}
        skip = ls.compute_skip_set(sims)
        assert 6 in skip

    def test_no_skip_layers_respected(self):
        ls = _ls(threshold=0.99)
        sims = {i: 0.999 for i in range(N_LAYERS)}
        skip = ls.compute_skip_set(sims)
        for i in range(NO_FIRST):
            assert i not in skip
        for i in range(N_LAYERS - NO_LAST, N_LAYERS):
            assert i not in skip

    def test_multiple_consecutive_skips(self):
        ls = _ls(threshold=0.99)
        sims = {5: 0.999, 6: 0.999, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 6 in skip
        assert 7 in skip
        assert 8 in skip

    def test_empty_similarities(self):
        ls = _ls()
        assert ls.compute_skip_set({}) == set()

    def test_unskippable_layer_blocks_candidate(self):
        ls = LayerSkip(LayerSkipConfig(no_skip_first=0, no_skip_last=0,
                                       threshold=0.99), num_layers=5)
        sims = {3: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 4 in skip

    # --- Contiguous gate ---

    def test_min_contiguous_3_passes(self):
        ls = _ls(threshold=0.99, min_contiguous=3)
        sims = {5: 0.999, 6: 0.999, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 8 in skip

    def test_min_contiguous_3_only_2_above(self):
        ls = _ls(threshold=0.99, min_contiguous=3)
        sims = {5: 0.980, 6: 0.999, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 8 not in skip

    def test_min_contiguous_3_gap_in_middle(self):
        ls = _ls(threshold=0.99, min_contiguous=3)
        sims = {5: 0.999, 6: 0.980, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 8 not in skip

    # --- Product gate ---

    def test_product_window_3_passes(self):
        ls = _ls(threshold=0.99, product_window=3, product_threshold=0.985)
        sims = {5: 0.996, 6: 0.997, 7: 0.998}
        skip = ls.compute_skip_set(sims)
        product = 0.996 * 0.997 * 0.998
        assert product >= 0.985
        assert 8 in skip

    def test_product_window_3_fails(self):
        ls = _ls(threshold=0.99, product_window=3, product_threshold=0.990)
        sims = {5: 0.996, 6: 0.990, 7: 0.998}
        skip = ls.compute_skip_set(sims)
        product = 0.996 * 0.990 * 0.998
        assert product < 0.990
        assert 8 not in skip

    # --- Both gates AND'd ---

    def test_contiguous_passes_product_fails(self):
        ls = _ls(threshold=0.99, min_contiguous=2,
                 product_window=3, product_threshold=0.995)
        sims = {5: 0.991, 6: 0.999, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 8 not in skip

    def test_both_gates_pass(self):
        ls = _ls(threshold=0.99, min_contiguous=2,
                 product_window=2, product_threshold=0.98)
        sims = {6: 0.999, 7: 0.999}
        skip = ls.compute_skip_set(sims)
        assert 8 in skip


# ---------------------------------------------------------------------------
# Is skippable
# ---------------------------------------------------------------------------


class TestIsSkippable:

    def test_first_layers_not_skippable(self):
        ls = _ls()
        for i in range(NO_FIRST):
            assert ls.is_skippable(i) is False

    def test_last_layers_not_skippable(self):
        ls = _ls()
        for i in range(N_LAYERS - NO_LAST, N_LAYERS):
            assert ls.is_skippable(i) is False

    def test_middle_layers_skippable(self):
        ls = _ls()
        for i in range(NO_FIRST, N_LAYERS - NO_LAST):
            assert ls.is_skippable(i) is True

    def test_boundary_values(self):
        ls = _ls()
        assert ls.is_skippable(NO_FIRST - 1) is False
        assert ls.is_skippable(NO_FIRST) is True
        assert ls.is_skippable(N_LAYERS - NO_LAST - 1) is True
        assert ls.is_skippable(N_LAYERS - NO_LAST) is False


# ---------------------------------------------------------------------------
# Should enable
# ---------------------------------------------------------------------------


class TestShouldEnable:

    def test_disabled_config(self):
        ls = _ls(enabled=False)
        assert ls.should_enable(1.0) is False

    def test_enabled_high_acceptance(self):
        ls = _ls(enabled=True, min_acceptance_rate=0.5)
        assert ls.should_enable(0.7) is True

    def test_enabled_low_acceptance(self):
        ls = _ls(enabled=True, min_acceptance_rate=0.5)
        assert ls.should_enable(0.3) is False

    def test_at_min_acceptance(self):
        ls = _ls(enabled=True, min_acceptance_rate=0.5)
        assert ls.should_enable(0.5) is True


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert LayerSkip.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert LayerSkip.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert LayerSkip.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_nearly_identical(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(128).astype(np.float32)
        noise = rng.standard_normal(128).astype(np.float32) * 0.001
        sim = LayerSkip.cosine_similarity(v, v + noise)
        assert sim > 0.999

    def test_zero_vector(self):
        v = np.array([1.0, 2.0], dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        assert LayerSkip.cosine_similarity(v, z) == 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_config(self):
        cfg = LayerSkipConfig()
        assert cfg.enabled is False
        assert cfg.threshold == pytest.approx(0.995)
        assert cfg.no_skip_first == 3
        assert cfg.no_skip_last == 3
        assert cfg.min_contiguous == 1
        assert cfg.product_window == 1
        assert cfg.product_threshold == pytest.approx(0.985)

    def test_frozen_config(self):
        cfg = LayerSkipConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]

    def test_is_enabled_property(self):
        assert LayerSkip(LayerSkipConfig(enabled=True)).is_enabled is True
        assert LayerSkip(LayerSkipConfig(enabled=False)).is_enabled is False

    def test_threshold_property(self):
        ls = LayerSkip(LayerSkipConfig(threshold=0.99))
        assert ls.threshold == pytest.approx(0.99)

    def test_no_skip_set_property(self):
        ls = _ls()
        nss = ls.no_skip_set
        assert 0 in nss
        assert 1 in nss
        assert 2 in nss
        assert N_LAYERS - 1 in nss
        assert N_LAYERS - 2 in nss
        assert N_LAYERS - 3 in nss
        assert 5 not in nss
