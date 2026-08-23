"""EPM-2 unit tests: tools/elb_train/metrics.py — coverage-aware
per-(position, layer) metrics on hand-computed small cases (exact values),
masked-cell exclusion, n reporting, serialization round trip."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import metrics  # noqa: E402


def _acc(g=2, layers=(3, 4), **kw):
    return metrics.EvalAccumulator(g, np.asarray(layers, np.int32), **kw)


class TestRecall:
    def test_top8_set_recall_exact(self):
        # One cell: pred top-8 hits exactly 5 of the true 8 -> 5/8.
        acc = _acc(g=1, layers=(3,))
        true = np.arange(8, dtype=np.int32).reshape(1, 1, 8)
        pred = np.array([[[0, 1, 2, 3, 4, 100, 101, 102]]], np.int32)
        acc.add_block(pred, true, np.array([True]))
        res = acc.result()
        assert res["recall"][8]["mean"][0, 0] == 5 / 8
        assert res["recall"][8]["n"][0, 0] == 1
        # m=16/32 not computable from an 8-wide prediction -> n stays 0.
        assert res["recall"][16]["n"][0, 0] == 0
        assert np.isnan(res["recall"][16]["mean"][0, 0])

    def test_recall_at_m_widths(self):
        # 32-wide ranked prediction: true ids sit at ranks 8..15 -> r@8=0,
        # r@16=1, r@32=1.  Exact hand case.
        acc = _acc(g=1, layers=(3,))
        true = (np.arange(8, dtype=np.int32) + 100).reshape(1, 1, 8)
        pred = np.concatenate([np.arange(8), np.arange(8) + 100,
                               np.arange(16) + 200]).astype(np.int32)
        acc.add_block(pred.reshape(1, 1, 32), true, np.array([True]))
        res = acc.result()
        assert res["recall"][8]["mean"][0, 0] == 0.0
        assert res["recall"][16]["mean"][0, 0] == 1.0
        assert res["recall"][32]["mean"][0, 0] == 1.0

    def test_masked_cells_excluded_and_n_reported(self):
        # Position 1 unlabeled: its (garbage) predictions must contribute
        # NOTHING; n stays 0 there and the mean is NaN.
        acc = _acc(g=2, layers=(3, 4))
        true = np.tile(np.arange(8, dtype=np.int32), (2, 2, 1))
        pred = np.tile(np.arange(8, dtype=np.int32), (2, 2, 1))
        pred[1] = 99  # would be recall 0 if (wrongly) counted
        acc.add_block(pred, true, np.array([True, False]))
        res = acc.result()
        assert res["recall"][8]["mean"][0, 0] == 1.0
        assert res["recall"][8]["n"][0, 0] == 1
        assert res["recall"][8]["n"][1, 0] == 0
        assert np.isnan(res["recall"][8]["mean"][1, 0])
        # Aggregate is over labeled cells only.
        v, n = metrics.overall_recall(res, 8)
        assert v == 1.0 and n == 2  # two layers at position 0

    def test_missing_prediction_excluded_and_counted(self):
        # All -1 prediction row = baseline had no source (e.g. B0 at the
        # first position): excluded from recall, counted in n_missing_pred.
        acc = _acc(g=1, layers=(3, 4))
        true = np.tile(np.arange(8, dtype=np.int32), (1, 2, 1))
        pred = np.tile(np.arange(8, dtype=np.int32), (1, 2, 1))
        pred[0, 1] = -1
        acc.add_block(pred, true, np.array([True]))
        res = acc.result()
        assert res["recall"][8]["n"][0, 0] == 1
        assert res["recall"][8]["n"][0, 1] == 0
        assert res["n_missing_pred"][0, 1] == 1
        assert res["n_missing_pred"][0, 0] == 0

    def test_true_pad_minus1_ignored_in_denominator(self):
        # True row with 4 valid ids (-1 pads): denominator is 4, not 8.
        acc = _acc(g=1, layers=(3,))
        true = np.array([[[5, 6, 7, 8, -1, -1, -1, -1]]], np.int32)
        pred = np.array([[[5, 6, 90, 91, 92, 93, 94, 95]]], np.int32)
        acc.add_block(pred, true, np.array([True]))
        res = acc.result()
        assert res["recall"][8]["mean"][0, 0] == 2 / 4


class TestUnionCoverage:
    def test_hand_computed_union(self):
        # Layer 3: pos0 pred {0..7}, pos1 pred {8..15}; true pos0 {0..7},
        # pos1 {16..23}.  Pred union {0..15}, true union {0..7, 16..23}
        # -> intersection {0..7} -> 8/16 = 0.5.
        acc = _acc(g=2, layers=(3,))
        true = np.stack([np.arange(8), np.arange(16, 24)]) \
                 .astype(np.int32).reshape(2, 1, 8)
        pred = np.stack([np.arange(8), np.arange(8, 16)]) \
                 .astype(np.int32).reshape(2, 1, 8)
        acc.add_block(pred, true, np.array([True, True]))
        res = acc.result()
        assert res["union_coverage"]["mean"][0] == 0.5
        assert res["union_coverage"]["n"][0] == 1

    def test_union_only_over_labeled_positions(self):
        acc = _acc(g=2, layers=(3,))
        true = np.tile(np.arange(8, dtype=np.int32), (2, 1, 1))
        pred = np.tile(np.arange(8, dtype=np.int32), (2, 1, 1))
        pred[1] = 99  # unlabeled -> must not pollute the union
        acc.add_block(pred, true, np.array([True, False]))
        res = acc.result()
        assert res["union_coverage"]["mean"][0] == 1.0


class TestCalibration:
    def test_two_bin_ece_exact(self):
        # 15 bins on [0,1].  Feed 4 predictions: two at score 0.95 (both
        # hits), two at score 0.05 (one hit) -> bins 14 and 0.
        # ECE = 0.5*|1-0.95| + 0.5*|0.5-0.05| = 0.25.
        acc = _acc(g=1, layers=(3,))
        true = np.arange(8, dtype=np.int32).reshape(1, 1, 8)
        pred = np.array([[[0, 1, 90, 91, -1, -1, -1, -1]]], np.int32)
        scores = np.array([[[0.95, 0.95, 0.05, 0.05,
                             np.nan, np.nan, np.nan, np.nan]]])
        # ids 0,1 hit; 90 misses; 91 misses... need one low-score hit:
        pred = np.array([[[0, 1, 2, 90, -1, -1, -1, -1]]], np.int32)
        scores = np.array([[[0.95, 0.95, 0.05, 0.05,
                             np.nan, np.nan, np.nan, np.nan]]])
        acc.add_block(pred, true, np.array([True]), pred_scores=scores)
        res = acc.result()
        cal = res["calibration"]
        assert cal["n"] == 4
        assert cal["bin_n"][14] == 2 and cal["bin_n"][0] == 2
        assert cal["bin_hit_rate"][14] == 1.0
        assert cal["bin_hit_rate"][0] == 0.5
        assert abs(cal["ece"] - (0.5 * 0.05 + 0.5 * 0.45)) < 1e-12

    def test_no_scores_no_calibration(self):
        acc = _acc(g=1, layers=(3,))
        true = np.arange(8, dtype=np.int32).reshape(1, 1, 8)
        acc.add_block(true.copy(), true, np.array([True]))
        assert acc.result()["calibration"]["n"] == 0
        assert np.isnan(acc.result()["calibration"]["ece"])


class TestAggregatesAndSerialization:
    def _small_result(self):
        acc = _acc(g=2, layers=(3, 4))
        true = np.tile(np.arange(8, dtype=np.int32), (2, 2, 1))
        pred = true.copy()
        pred[0, 1, :4] = 100  # half wrong at (0, layer 4)
        sc = np.full((2, 2, 8), 0.9)
        acc.add_block(pred, true, np.array([True, True]), pred_scores=sc)
        return acc.result()

    def test_overall_and_per_position(self):
        res = self._small_result()
        v, n = metrics.overall_recall(res, 8)
        assert n == 4
        assert abs(v - (1.0 + 0.5 + 1.0 + 1.0) / 4) < 1e-12
        rows = metrics.per_position_recall(res, 8)
        assert rows[0]["n"] == 2 and abs(rows[0]["recall"] - 0.75) < 1e-12
        assert rows[1]["n"] == 2 and rows[1]["recall"] == 1.0

    def test_wmap_weighted_recall_exact(self):
        # Cells: (k0,L3)=1.0 (k0,L4)=0.5 (k1,L3)=1.0 (k1,L4)=1.0, all n=1.
        # W = [[4, 2], [1, 1]] -> (4*1 + 2*0.5 + 1 + 1) / 8 = 7/8.
        res = self._small_result()
        w = np.array([[4.0, 2.0], [1.0, 1.0]])
        v, cov = metrics.wmap_weighted_recall(res, w, 8)
        assert abs(v - 7 / 8) < 1e-12
        assert cov == 1.0

    def test_wmap_weighted_recall_masked_cells_renormalize(self):
        # Kill (k1, *) labels: only k0 cells observed -> renormalized over
        # W[0] = [4, 2] -> (4*1 + 2*0.5)/6 = 5/6; coverage = 6/8.
        acc = _acc()
        true = np.tile(np.arange(8, dtype=np.int32), (2, 2, 1))
        pred = true.copy()
        pred[0, 1, :4] = 100
        acc.add_block(pred, true, np.array([True, False]))
        w = np.array([[4.0, 2.0], [1.0, 1.0]])
        v, cov = metrics.wmap_weighted_recall(acc.result(), w, 8)
        assert abs(v - 5 / 6) < 1e-12
        assert abs(cov - 6 / 8) < 1e-12

    def test_wmap_weighted_recall_shape_guard(self):
        res = self._small_result()
        try:
            metrics.wmap_weighted_recall(res, np.ones((3, 2)), 8)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_save_load_round_trip(self, tmp_path):
        res = self._small_result()
        metrics.save_result(res, tmp_path / "m", meta={"tag": "t"})
        assert (tmp_path / "m.npz").is_file()
        with open(tmp_path / "m.json", encoding="utf-8") as f:
            js = json.load(f)
        assert js["meta"]["tag"] == "t"
        assert abs(js["overall_recall8"] - 0.875) < 1e-12
        back = metrics.load_result_npz(tmp_path / "m")
        np.testing.assert_allclose(back["recall"][8]["mean"],
                                   res["recall"][8]["mean"])
        np.testing.assert_array_equal(back["recall"][8]["n"],
                                      res["recall"][8]["n"])
        np.testing.assert_allclose(back["union_coverage"]["mean"],
                                   res["union_coverage"]["mean"])
        assert abs(back["calibration"]["ece"]
                   - res["calibration"]["ece"]) < 1e-12

    def test_summary_table_renders_n(self):
        res = self._small_result()
        md = metrics.summary_table({"base": res})
        assert "| base |" in md
        assert "(4)" in md  # n annotated

    def test_heatmap_skips_gracefully_without_matplotlib(self, tmp_path):
        res = self._small_result()
        ok = metrics.save_heatmap_png(
            res["recall"][8]["mean"], res["recall"][8]["n"],
            tmp_path / "h.png", "t", res["moe_layers"])
        # Either matplotlib exists and wrote the file, or it returned False
        # and wrote nothing — never raises.
        assert ok == (tmp_path / "h.png").is_file()

    def test_determinism(self):
        a = self._small_result()
        b = self._small_result()
        np.testing.assert_array_equal(a["recall"][8]["mean"],
                                      b["recall"][8]["mean"])
        assert a["calibration"]["ece"] == b["calibration"]["ece"]
