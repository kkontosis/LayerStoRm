"""P11 Stage-0 miss-mass decomposition + score-drift sigma estimation
(spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(6)).

Per committed decode position t >= 1 and layer j, the b0_prev prediction
is top8(t-1) and the miss set is  Miss(t,j) = top8(t) \\ top8(t-1).
Each missed expert e is classified with EXPLICIT PRECEDENCE (so the three
classes partition Miss exactly):

  1. BOUNDARY   margin_e(t) = g_e(t) - g_rank9(t) < c * sigma_j
                — e only just made the true top-8 cut; whether it is in
                or out is within score-drift noise, irreducible for ANY
                predictor.  g is the noaux_tc SELECTION score
                (sigmoid(logits) + correction bias, glm_router).
  2. WITHIN-POOL  e in RecentUnion_W(t,j) — history-addressable; the
                ceiling of any history-based (score-tracker) model.
  3. NOVEL      outside the recent union — needs content features.

Windows W: prevpos (top8(t-1) itself — degenerate control, f_within == 0
by construction), prevchunk (previous verify chunk's union), trail16 /
trail64 (trailing committed positions, strictly before t).

sigma_j: MAD-robust std (1.4826 * MAD) of consecutive-position deltas
g_e(t) - g_e(t-1), restricted to experts in the top-32 of either
position (the boundary-relevant region — the never-active floor would
otherwise dominate), estimated on the TRAIN split.
"""

from __future__ import annotations

import dataclasses

import numpy as np

MAD_TO_STD = 1.4826


# ── sigma estimation ─────────────────────────────────────────────────────────

class SigmaEstimator:
    """Per-layer reservoir of |top-32-restricted| consecutive score deltas.
    Deterministic stride thinning keeps <= cap samples per layer."""

    def __init__(self, n_layers: int, cap: int = 200_000):
        self.cap = cap
        self.buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.count = np.zeros(n_layers, np.int64)   # total seen (pre-thin)

    def add_sequence_layer(self, j: int, sel: np.ndarray) -> None:
        """sel: [T, E] float32 selection scores of one sequence, layer j."""
        if sel.shape[0] < 2:
            return
        e32 = sel.shape[1] - 32
        thr = np.partition(sel, e32, axis=1)[:, e32]      # 32nd-largest
        top32 = sel >= thr[:, None]
        pair = top32[1:] | top32[:-1]
        d = (sel[1:] - sel[:-1])[pair]
        self.count[j] += d.size
        # thin to keep the reservoir bounded (deterministic stride).
        have = sum(b.size for b in self.buf[j])
        room = self.cap - have
        if room <= 0:
            return
        if d.size > room:
            d = d[:: int(np.ceil(d.size / room))]
        self.buf[j].append(d.astype(np.float32))

    def sigma(self) -> np.ndarray:
        """MAD-robust std per layer; NaN where no samples."""
        out = np.full(len(self.buf), np.nan, np.float32)
        for j, chunks in enumerate(self.buf):
            if not chunks:
                continue
            d = np.concatenate(chunks)
            med = np.median(d)
            out[j] = MAD_TO_STD * np.median(np.abs(d - med))
        return out


# ── decomposition ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DecompAccum:
    """Accumulators over (c, window, layer). Slot counts and gate-weight
    sums; boundary is window-independent, within/novel are per window."""
    windows: list[str]
    cs: list[float]
    n_layers: int

    def __post_init__(self):
        C, W, J = len(self.cs), len(self.windows), self.n_layers
        self.n_true = np.zeros(J, np.int64)      # evaluated true slots
        self.n_hit = np.zeros(J, np.int64)       # b0_prev hits
        self.n_miss = np.zeros(J, np.int64)
        self.w_true = np.zeros(J, np.float64)    # gate-weight versions
        self.w_hit = np.zeros(J, np.float64)
        self.w_miss = np.zeros(J, np.float64)
        self.n_boundary = np.zeros((C, J), np.int64)
        self.w_boundary = np.zeros((C, J), np.float64)
        self.n_within = np.zeros((C, W, J), np.int64)
        self.w_within = np.zeros((C, W, J), np.float64)
        self.n_novel = np.zeros((C, W, J), np.int64)
        self.w_novel = np.zeros((C, W, J), np.float64)

    def add_sequence_layer(self, j: int, tops: np.ndarray, w: np.ndarray,
                           sel: np.ndarray, sigma_j: float,
                           chunk_id: np.ndarray) -> None:
        """One sequence, one layer.
        tops [T, K] int32 true top-8; w [T, K] f32 gate weights;
        sel [T, E] f32 selection scores; chunk_id [T] int64 (verify-chunk
        ordinal within the sequence, non-decreasing)."""
        T, K = tops.shape
        if T < 2:
            return
        E = sel.shape[1]
        routed = np.zeros((T, E), np.uint8)
        np.put_along_axis(routed, tops.astype(np.int64), 1, axis=1)
        # b0_prev miss mask over evaluated positions t >= 1.
        prev_routed = routed[:-1]                          # [T-1, E]
        cur = tops[1:].astype(np.int64)                    # [T-1, K]
        cur_w = w[1:]
        miss = np.take_along_axis(prev_routed, cur, axis=1) == 0
        self.n_true[j] += miss.size
        self.n_hit[j] += int((~miss).sum())
        self.n_miss[j] += int(miss.sum())
        self.w_true[j] += float(cur_w.sum())
        self.w_hit[j] += float(cur_w[~miss].sum())
        self.w_miss[j] += float(cur_w[miss].sum())

        # margins of the true top-K vs the rank-(K+1) score (K=8 in the
        # corpus; kept general so tiny test fixtures work).
        rk1 = np.partition(sel[1:], E - K - 1, axis=1)[:, E - K - 1]
        margin = np.take_along_axis(sel[1:], cur, axis=1) - rk1[:, None]

        # window membership of each (t, slot): expert in RecentUnion_W.
        csum = np.zeros((T + 1, E), np.int32)
        np.cumsum(routed, axis=0, out=csum[1:])
        t_idx = np.arange(1, T)

        def trail_member(h: int) -> np.ndarray:
            lo = np.maximum(t_idx - h, 0)
            cnt = (csum[t_idx.reshape(-1, 1), cur]
                   - csum[lo.reshape(-1, 1), cur])
            return cnt > 0

        memberships = []
        for name in self.windows:
            if name == "prevpos":
                memberships.append(np.take_along_axis(
                    prev_routed, cur, axis=1) > 0)         # == ~miss
            elif name == "trail16":
                memberships.append(trail_member(16))
            elif name == "trail64":
                memberships.append(trail_member(64))
            elif name == "prevchunk":
                n_chunks = int(chunk_id[-1]) + 1
                occ_chunk = np.zeros((n_chunks, E), np.uint8)
                np.maximum.at(occ_chunk, chunk_id, routed)
                prev_chunk = chunk_id[1:] - 1              # [T-1]
                m = np.zeros_like(miss)
                ok = prev_chunk >= 0
                if np.any(ok):
                    m[ok] = np.take_along_axis(
                        occ_chunk[prev_chunk[ok]], cur[ok], axis=1) > 0
                memberships.append(m)
            else:
                raise ValueError(f"unknown window {name!r}")

        for ci, cval in enumerate(self.cs):
            bnd = miss & (margin < cval * sigma_j)
            self.n_boundary[ci, j] += int(bnd.sum())
            self.w_boundary[ci, j] += float(cur_w[bnd].sum())
            rest = miss & ~bnd
            for wi, mem in enumerate(memberships):
                within = rest & mem
                novel = rest & ~mem
                self.n_within[ci, wi, j] += int(within.sum())
                self.w_within[ci, wi, j] += float(cur_w[within].sum())
                self.n_novel[ci, wi, j] += int(novel.sum())
                self.w_novel[ci, wi, j] += float(cur_w[novel].sum())

    # ── reporting ────────────────────────────────────────────────────────

    def summarize(self, moe_layers: np.ndarray, deep_lo: int,
                  deep_hi: int, sigma: np.ndarray) -> dict:
        """Aggregate fractions per (c, window) for ALL and deep layers +
        per-layer ceiling curves. Partition identity asserted."""
        deep = np.array([i for i, l in enumerate(moe_layers)
                         if deep_lo <= l <= deep_hi])
        assert np.all(self.n_boundary[:, None, :] + self.n_within
                      + self.n_novel == self.n_miss[None, None, :]), \
            "decomposition does not partition the miss mass"

        def agg(idx) -> dict:
            nt = self.n_true[idx].sum()
            nm = self.n_miss[idx].sum()
            wt = self.w_true[idx].sum()
            out = {
                "recall8": float(self.n_hit[idx].sum() / max(1, nt)),
                "miss_mass": float(nm / max(1, nt)),
                "recall8_w": float(self.w_hit[idx].sum() / max(1e-12, wt)),
                "grid": {},
            }
            for ci, cval in enumerate(self.cs):
                nb = self.n_boundary[ci, idx].sum()
                for wi, wname in enumerate(self.windows):
                    nw = self.n_within[ci, wi, idx].sum()
                    nn = self.n_novel[ci, wi, idx].sum()
                    ww = self.w_within[ci, wi, idx].sum()
                    wn = self.w_novel[ci, wi, idx].sum()
                    wb = self.w_boundary[ci, idx].sum()
                    key = f"c{cval:g}|{wname}"
                    out["grid"][key] = {
                        "f_boundary": float(nb / max(1, nm)),
                        "f_within": float(nw / max(1, nm)),
                        "f_novel": float(nn / max(1, nm)),
                        "f_within_w": float(ww / max(1e-12, ww + wn + wb)),
                        # absolute recall@8 headroom of a history model
                        "within_headroom": float(nw / max(1, nt)),
                    }
            return out

        per_layer_ceiling = {}
        for ci, cval in enumerate(self.cs):
            for wi, wname in enumerate(self.windows):
                r = self.n_hit / np.maximum(1, self.n_true)
                ceil = r + self.n_within[ci, wi] / np.maximum(1, self.n_true)
                per_layer_ceiling[f"c{cval:g}|{wname}"] = \
                    np.round(ceil, 4).tolist()
        return {
            "windows": self.windows, "cs": self.cs,
            "sigma_per_layer": np.round(sigma, 5).tolist(),
            "all": agg(slice(None)),
            "deep": agg(deep),
            "deep_layer_ids": [int(moe_layers[i]) for i in deep],
            "per_layer_recall8": np.round(
                self.n_hit / np.maximum(1, self.n_true), 4).tolist(),
            "per_layer_ceiling": per_layer_ceiling,
        }
