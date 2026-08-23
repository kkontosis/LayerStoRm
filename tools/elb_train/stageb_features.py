"""P11.b feature engine: online-causal per-candidate features for the
RESIDUAL probe protocol (spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(4);
Stage-0 verdict studies/epm/p11-stage0/summary.md).

Framing (§3-P11(2)): the addressable mass is WITHIN-POOL — next top-8
members already inside the recent union. Candidates at (t, j) are the
trail-64 union (prev-top8 is a subset after one step); the probe's job is
to rank them, i.e. learn the correction ON TOP of the free history
baselines. All features are computed from state BEFORE the position's
routing is revealed (predict-then-update — same causality contract as
signal_bank).

Feature classes (per-class probes kill dead classes, §3-P11(4)(i)):
  temporal    in_prev, trail16, trail64, recency
  transition  trans (transition-EMA score), trans_rank (rank among the
              candidates — scale-free)
  score       ema2, ema8 (sel-space EMAs), g_last (last sel score),
              margin_last (signed distance to last rank-8 cut — the
              boundary-proximity signal), drift1 (last sel delta)
  position    layer_f (j / J)

Boundary stratification (report-only, NEVER a feature): label-side
margin at t classifies each true-slot row as boundary (margin < c*sigma_j)
so probe wins can be split into addressable vs noise-fitting mass.
"""

from __future__ import annotations

import numpy as np

from . import signal_bank

FEATURES = ("in_prev", "trail16", "trail64", "recency",
            "trans", "trans_rank",
            "ema2", "ema8", "g_last", "margin_last", "drift1",
            "layer_f")
CLASSES = {
    "temporal": ("in_prev", "trail16", "trail64", "recency"),
    "transition": ("trans", "trans_rank"),
    "score": ("ema2", "ema8", "g_last", "margin_last", "drift1"),
    "position": ("layer_f",),
}
F_IDX = {f: i for i, f in enumerate(FEATURES)}
RECENCY_CAP = 512.0


class HistoryState:
    """Per-sequence online state over all layers. Global (cross-sequence)
    state: the transition-EMA table — deployment accumulates it across
    requests (labels are free at every verify); everything else resets."""

    def __init__(self, n_layers: int, n_experts: int):
        self.J = n_layers
        self.E = n_experts
        self.trans = signal_bank.TransEma(n_layers, n_experts)
        self.trail16 = signal_bank.TrailFreq(n_layers, n_experts, 16)
        self.trail64 = signal_bank.TrailFreq(n_layers, n_experts, 64)
        self.ema2 = signal_bank.ScoreEma(n_layers, n_experts, 2.0)
        self.ema8 = signal_bank.ScoreEma(n_layers, n_experts, 8.0)
        self.prev8 = np.zeros((n_layers, n_experts), np.float32)
        self.g_prev = np.zeros((n_layers, n_experts), np.float32)
        self.g_prev2 = np.zeros((n_layers, n_experts), np.float32)
        self.cut_prev = np.zeros(n_layers, np.float32)
        self.last_seen = np.full((n_layers, n_experts), -1, np.int64)
        self.t_idx = 0
        self._j_arange = np.arange(n_layers)
        self._cache_t = -1
        self._cache: dict = {}

    def reset_seq(self) -> None:
        for m in (self.trans, self.trail16, self.trail64,
                  self.ema2, self.ema8):
            m.reset_seq()
        self.prev8[:] = 0.0
        self.g_prev[:] = 0.0
        self.g_prev2[:] = 0.0
        self.cut_prev[:] = 0.0
        self.last_seen[:] = -1
        self.t_idx = 0

    @property
    def ready(self) -> bool:
        """Features are meaningful only from the second position on."""
        return self.t_idx >= 1

    def _frame(self) -> dict:
        """Per-position cache of the all-layer member predictions (the
        members compute [J, E] arrays; per-layer callers must not pay
        that J times)."""
        if self._cache_t != self.t_idx:
            # trans raw table values grow ~1/scale between lazy-decay
            # renormalizations (rank-safe, regression-hostile); multiply
            # by the scale to get stationary decayed-count units.
            self._cache = {
                "trans": self.trans.predict(0) * self.trans.scale,
                "ema2": self.ema2.predict(0),
                "ema8": self.ema8.predict(0)}
            self._cache_t = self.t_idx
        return self._cache

    def candidates(self, j: int) -> np.ndarray:
        """int64 candidate expert ids at layer j = trail-64 union
        (prev-top8 is a subset — routed at t-1 implies in the window)."""
        return np.nonzero(self.trail64.counts[j] > 0)[0].astype(np.int64)

    def feature_block(self, j: int, cand: np.ndarray) -> np.ndarray:
        """X [len(cand), F] float32 from PRE-update state."""
        n = len(cand)
        X = np.empty((n, len(FEATURES)), np.float32)
        frame = self._frame()
        rec = np.where(self.last_seen[j, cand] >= 0,
                       self.t_idx - self.last_seen[j, cand],
                       RECENCY_CAP).astype(np.float32)
        tr = frame["trans"][j, cand]
        # scale-free rank of the transition score among the candidates
        order = np.argsort(np.argsort(-tr, kind="stable"), kind="stable")
        X[:, F_IDX["in_prev"]] = self.prev8[j, cand]
        X[:, F_IDX["trail16"]] = self.trail16.counts[j, cand]
        X[:, F_IDX["trail64"]] = self.trail64.counts[j, cand]
        X[:, F_IDX["recency"]] = np.minimum(rec, RECENCY_CAP)
        X[:, F_IDX["trans"]] = tr
        X[:, F_IDX["trans_rank"]] = (order / max(1, n - 1)).astype(
            np.float32)
        X[:, F_IDX["ema2"]] = frame["ema2"][j, cand]
        X[:, F_IDX["ema8"]] = frame["ema8"][j, cand]
        X[:, F_IDX["g_last"]] = self.g_prev[j, cand]
        X[:, F_IDX["margin_last"]] = (self.g_prev[j, cand]
                                      - self.cut_prev[j])
        X[:, F_IDX["drift1"]] = (self.g_prev[j, cand]
                                 - self.g_prev2[j, cand])
        X[:, F_IDX["layer_f"]] = j / max(1, self.J - 1)
        return X

    def update(self, tops: np.ndarray, top_w: np.ndarray,
               sel: np.ndarray) -> None:
        """Reveal position t's routing: tops [J, K] int32, top_w [J, K],
        sel [J, E] float32 selection scores."""
        t64 = tops.astype(np.int64)
        for m in (self.trans, self.trail16, self.trail64,
                  self.ema2, self.ema8):
            m.update(tops, top_w, sel, 0)
        self.prev8[:] = 0.0
        np.put_along_axis(self.prev8, t64, 1.0, axis=1)
        self.g_prev2[:] = self.g_prev
        self.g_prev[:] = sel
        K = tops.shape[1]
        self.cut_prev[:] = np.partition(sel, self.E - K,
                                        axis=1)[:, self.E - K]
        self.last_seen[self._j_arange[:, None], t64] = self.t_idx
        self.t_idx += 1


def label_rows(cand: np.ndarray, tops_j: np.ndarray, sel_j: np.ndarray,
               sigma_j: float, c: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-side quantities for the candidate rows at (t, j):
      y            int8  — candidate is in the true top-K at t
      is_miss      int8  — y AND not in the true top-K at t-1 is decided
                           by the caller via the in_prev feature; here
                           miss = y (caller masks with in_prev)
      is_boundary  int8  — y AND margin_t < c*sigma_j (report-only strat)
    """
    K = len(tops_j)
    E = len(sel_j)
    member = np.zeros(E, bool)
    member[tops_j.astype(np.int64)] = True
    y = member[cand].astype(np.int8)
    cut9 = np.partition(sel_j, E - K - 1)[E - K - 1]
    margin = sel_j[cand] - cut9
    bnd = (y > 0) & (margin < c * sigma_j)
    return y, margin.astype(np.float32), bnd.astype(np.int8)
