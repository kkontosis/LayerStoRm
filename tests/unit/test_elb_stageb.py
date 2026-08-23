"""P11.b feature-engine tests: online causality + hand-computed features
(tools/elb_train/stageb_features.py)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "tools"))

from elb_train import stageb_features as sf  # noqa: E402


def _step(hs, tops, sel):
    w = np.ones_like(tops, np.float32)
    hs.update(np.asarray(tops, np.int32), w,
              np.asarray(sel, np.float32))


class TestHistoryState:
    J, E = 2, 12

    def test_candidates_and_hand_features(self):
        hs = sf.HistoryState(self.J, self.E)
        sel0 = np.zeros((self.J, self.E), np.float32)
        sel0[0, [3, 7]] = [0.9, 0.8]
        _step(hs, [[3, 7], [1, 2]], sel0)
        sel1 = np.zeros((self.J, self.E), np.float32)
        sel1[0, [3, 5]] = [0.85, 0.7]
        _step(hs, [[3, 5], [1, 4]], sel1)
        assert hs.ready
        cand = hs.candidates(0)
        assert set(cand.tolist()) == {3, 5, 7}
        X = hs.feature_block(0, cand)
        row = {int(e): X[i] for i, e in enumerate(cand)}
        fi = sf.F_IDX
        # expert 3: in prev top8, trail64 count 2, seen at t=1
        assert row[3][fi["in_prev"]] == 1.0
        assert row[3][fi["trail64"]] == 2.0
        assert row[3][fi["recency"]] == 1.0        # t_idx 2 - last_seen 1
        # expert 7: only at t=0 -> not in prev, recency 2
        assert row[7][fi["in_prev"]] == 0.0
        assert row[7][fi["trail64"]] == 1.0
        assert row[7][fi["recency"]] == 2.0
        # g_last / margin_last from sel1: K=2 cut = 2nd largest = 0.7
        assert abs(row[3][fi["g_last"]] - 0.85) < 1e-6
        assert abs(row[3][fi["margin_last"]] - 0.15) < 1e-6
        assert abs(row[5][fi["margin_last"]] - 0.0) < 1e-6
        # drift1 for expert 3: 0.85 - 0.9
        assert abs(row[3][fi["drift1"]] + 0.05) < 1e-6

    def test_predict_then_update_causality(self):
        rng = np.random.default_rng(0)
        hs = sf.HistoryState(self.J, self.E)
        tops = rng.integers(0, self.E, (6, self.J, 2))
        sels = rng.random((6, self.J, self.E)).astype(np.float32)
        feats = []
        for t in range(6):
            if hs.ready:
                c = hs.candidates(0)
                feats.append((c.copy(), hs.feature_block(0, c).copy()))
            _step(hs, tops[t], sels[t])
        # replay truncated stream: identical prefix features
        hs2 = sf.HistoryState(self.J, self.E)
        feats2 = []
        for t in range(4):
            if hs2.ready:
                c = hs2.candidates(0)
                feats2.append((c.copy(), hs2.feature_block(0, c).copy()))
            _step(hs2, tops[t], sels[t])
        for (c1, x1), (c2, x2) in zip(feats2, feats[: len(feats2)]):
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_array_equal(x1, x2)

    def test_reset_clears_per_seq_not_trans_table(self):
        hs = sf.HistoryState(self.J, self.E)
        sel = np.random.default_rng(1).random(
            (self.J, self.E)).astype(np.float32)
        _step(hs, [[0, 1], [2, 3]], sel)
        _step(hs, [[4, 5], [6, 7]], sel)     # trans learns (0,1)->(4,5)
        table_sum = hs.trans.table.sum()
        assert table_sum > 0
        hs.reset_seq()
        assert not hs.ready
        assert hs.trail64.counts.sum() == 0
        assert hs.g_prev.sum() == 0
        assert hs.trans.table.sum() == table_sum   # global state kept

    def test_label_rows(self):
        sel = np.zeros(12, np.float32)
        sel[[3, 5]] = [0.9, 0.2]
        cand = np.array([3, 5, 7], np.int64)
        tops = np.array([3, 5], np.int32)
        y, margin, bnd = sf.label_rows(cand, tops, sel, sigma_j=0.3, c=1.0)
        np.testing.assert_array_equal(y, [1, 1, 0])
        # cut9 (K=2 -> rank-3 value) = 0.0; margins 0.9, 0.2, 0.0
        assert abs(margin[0] - 0.9) < 1e-6
        np.testing.assert_array_equal(bnd, [0, 1, 0])  # 0.2 < 0.3


# ── stagec: online mixture primitives ────────────────────────────────────────

class TestStagec:
    def test_online_ridge_recovers_linear(self):
        from elb_train.stagec import OnlineRidge
        rng = np.random.default_rng(0)
        d = 3
        w_true = np.array([2.0, -1.0, 0.5])
        r = OnlineRidge(1, d=d + 1, decay=1.0, lam=1e-6, refresh=10)
        for _ in range(30):
            X = rng.random((20, d))
            y = X @ w_true + 0.3
            r.update(0, X, y)
            r.tick()
        np.testing.assert_allclose(r.w[0][:d], w_true, atol=1e-3)
        assert abs(r.w[0][d] - 0.3) < 1e-3          # bias
        # prediction uses the solved weights
        X = rng.random((5, d))
        np.testing.assert_allclose(r.predict(0, X), X @ w_true + 0.3,
                                   atol=1e-3)

    def test_hedge_converges_to_best_member(self):
        from elb_train.stagec import HedgeMixer
        rng = np.random.default_rng(1)
        h = HedgeMixer(1, 3, eta=0.2)
        for _ in range(300):
            # member 1 consistently best (lowest loss)
            losses = np.array([0.8, 0.2, 0.6]) + rng.random(3) * 0.05
            h.update(0, losses)
        assert h.w[0, 1] > 0.95
        assert abs(h.w[0].sum() - 1.0) < 1e-9

    def test_rank_norm(self):
        from elb_train.stagec import _rank_norm
        sc = np.array([0.1, 5.0, -2.0, 3.0])
        rn = _rank_norm(sc)
        assert rn[1] == 1.0 and rn[2] == 0.0
        assert rn[3] > rn[0]
        assert len(_rank_norm(np.array([1.0]))) == 1
