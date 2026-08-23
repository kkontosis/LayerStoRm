"""EPM-2 coverage-aware evaluation metrics (Phase 29, MoE-SpeQ_NOTES.md §3/§7).

All metrics are per-(position k, layer j) cell and COVERAGE-AWARE: under
sequential early-stop verification only the verified prefix of a block has
labels (TD-EPM-REJECTED-LABELS), so label coverage decays with position
~a_j.  Every aggregate therefore carries its sample count n and NEVER
averages over masked cells (masked cells contribute neither numerator nor
denominator; empty cells report NaN mean + n=0).

Acceptance-conditioned variants are INTRINSIC here: label_mask is exactly
"this position was verified", so every metric this module emits is already
the acceptance-conditioned variant — there is no separate unconditioned
family to compute until rejected-suffix labels exist (TD-EPM-REJECTED-LABELS
option (b)).

Metrics (predictions are ranked expert-id lists, best first):
  recall@m (m in {8,16,32}):  |pred_top_m ∩ true_top8| / |true_top8| per
      labeled (k, j) cell.  m=8 is "top-8 set recall", the primary metric.
      Only computed for m <= the ranked width the baseline provides (B0
      provides 8 ids -> recall@16/32 absent, reported as n=0).
  union coverage per (block, j): |∪_k pred_top8(k) ∩ ∪_k true_top8(k)| /
      |∪_k true_top8(k)| over the block's labeled positions; positions with
      a missing prediction are dropped from BOTH unions (data-boundary
      artifact, not a model miss); aggregated per layer.
  score calibration: reliability of predicted scores vs the hit indicator
      (predicted expert ∈ true top-8) over fixed bins on [0, 1] (scores
      clipped) + ECE = Σ_b (n_b/N) |hit_rate_b − mean_score_b|.  Meaningful
      as calibration only for probability-like scores (B1/B2 sigmoid
      scores); for B0's renormalized gate-weight proxy it is a monotonicity
      diagnostic — read it with that caveat.
  missing predictions: cells where the baseline could not predict at all
      (e.g. B0 at a sequence's first position) are EXCLUDED from recall and
      counted in n_missing_pred [G, J].

Serialized artifact contract (EPM-3 consumes this):
  save_result(result, prefix) writes
    <prefix>.npz  — recall8/recall16/recall32          f64 [G, J] (NaN=empty)
                    n8/n16/n32                          i64 [G, J]
                    n_missing_pred                      i64 [G, J]
                    union_coverage                      f64 [J]
                    union_n                             i64 [J]
                    calib_bin_edges [B+1], calib_bin_n [B],
                    calib_bin_mean_score [B], calib_bin_hit_rate [B]
                    moe_layers                          i32 [J]
    <prefix>.json — scalar aggregates: ece, calib_n, overall (n-weighted)
                    recall@m, per-position recall@8 rows, provenance.
  Heatmap PNGs (position × layer) are written when matplotlib is
  importable, silently skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RECALL_MS = (8, 16, 32)


class EvalAccumulator:
    """Streaming per-(k, j) metric accumulator for one baseline variant.

    Feed blocks with add_block(); read the aggregate with result().
    Deterministic: pure sums, no RNG.
    """

    def __init__(self, max_gamma: int, moe_layers: np.ndarray | list[int],
                 ms: tuple[int, ...] = RECALL_MS, n_calib_bins: int = 15,
                 topk: int = 8) -> None:
        self.g = int(max_gamma)
        self.moe_layers = np.asarray(moe_layers, np.int32)
        self.j = len(self.moe_layers)
        self.ms = tuple(sorted(ms))
        self.topk = int(topk)
        self._sum = {m: np.zeros((self.g, self.j)) for m in self.ms}
        self._n = {m: np.zeros((self.g, self.j), np.int64) for m in self.ms}
        self._missing = np.zeros((self.g, self.j), np.int64)
        self._union_sum = np.zeros(self.j)
        self._union_n = np.zeros(self.j, np.int64)
        self._nb = int(n_calib_bins)
        self._cal_edges = np.linspace(0.0, 1.0, self._nb + 1)
        self._cal_n = np.zeros(self._nb, np.int64)
        self._cal_score = np.zeros(self._nb)
        self._cal_hit = np.zeros(self._nb)

    # ── feeding ──────────────────────────────────────────────────────────

    def add_block(self, pred_ids: np.ndarray, true_ids: np.ndarray,
                  label_mask: np.ndarray,
                  pred_scores: np.ndarray | None = None) -> None:
        """One block's predictions vs labels.

        pred_ids:    int [Gb, J, M] ranked ids, best first; a ROW of all -1
                     = prediction unavailable for that (k, j) cell (counted
                     in n_missing_pred, excluded from recall/union); -1
                     entries inside a row are empty slots (ignored).
        true_ids:    int [Gb, J, K] true top-8 ids (-1 pad).
        label_mask:  bool [Gb] — only True positions contribute anything.
        pred_scores: float [Gb, J, M] scores aligned with pred_ids (optional;
                     feeds calibration; NaN entries skipped).
        """
        pred_ids = np.asarray(pred_ids)
        true_ids = np.asarray(true_ids)
        label_mask = np.asarray(label_mask, bool)
        gb = pred_ids.shape[0]
        if true_ids.shape[0] != gb or label_mask.shape[0] != gb:
            raise ValueError("pred/true/mask leading dims disagree")
        if pred_ids.shape[1] != self.j or true_ids.shape[1] != self.j:
            raise ValueError("layer dim mismatch with moe_layers")
        m_avail = pred_ids.shape[2]

        # Per-(block,j) unions over labeled positions.
        union_pred: list[set] = [set() for _ in range(self.j)]
        union_true: list[set] = [set() for _ in range(self.j)]
        union_valid = np.zeros(self.j, bool)

        for k in range(min(gb, self.g)):
            if not label_mask[k]:
                continue
            for j in range(self.j):
                t = true_ids[k, j]
                t = t[t >= 0]
                if t.size == 0:
                    continue  # labeled flag but empty truth: skip cell
                p = pred_ids[k, j]
                if np.all(p < 0):
                    self._missing[k, j] += 1
                    continue
                tset = set(int(x) for x in t)
                for m in self.ms:
                    if m > m_avail:
                        continue
                    pm = p[:m]
                    pm = pm[pm >= 0]
                    hits = len(tset.intersection(int(x) for x in pm))
                    self._sum[m][k, j] += hits / len(tset)
                    self._n[m][k, j] += 1
                # Unions use the top-topk prediction slice.
                pk = p[:self.topk]
                union_pred[j].update(int(x) for x in pk[pk >= 0])
                union_true[j].update(tset)
                union_valid[j] = True
                # Calibration pairs.
                if pred_scores is not None:
                    sc = np.asarray(pred_scores[k, j], np.float64)
                    for slot in range(min(m_avail, len(sc))):
                        e = int(p[slot])
                        s = sc[slot]
                        if e < 0 or not np.isfinite(s):
                            continue
                        b = min(int(np.clip(s, 0.0, 1.0) * self._nb),
                                self._nb - 1)
                        self._cal_n[b] += 1
                        self._cal_score[b] += float(np.clip(s, 0.0, 1.0))
                        self._cal_hit[b] += 1.0 if e in tset else 0.0

        for j in range(self.j):
            if union_valid[j] and union_true[j]:
                inter = len(union_pred[j] & union_true[j])
                self._union_sum[j] += inter / len(union_true[j])
                self._union_n[j] += 1

    # ── results ──────────────────────────────────────────────────────────

    def result(self) -> dict:
        out: dict = {"moe_layers": self.moe_layers,
                     "max_gamma": self.g, "recall": {}, }
        for m in self.ms:
            n = self._n[m]
            mean = np.where(n > 0, self._sum[m] / np.maximum(n, 1), np.nan)
            out["recall"][m] = {"mean": mean, "n": n.copy()}
        out["n_missing_pred"] = self._missing.copy()
        un = self._union_n
        out["union_coverage"] = {
            "mean": np.where(un > 0, self._union_sum / np.maximum(un, 1),
                             np.nan),
            "n": un.copy()}
        n = self._cal_n
        tot = int(n.sum())
        mean_s = np.where(n > 0, self._cal_score / np.maximum(n, 1), np.nan)
        hit_r = np.where(n > 0, self._cal_hit / np.maximum(n, 1), np.nan)
        ece = (float(np.sum(np.abs(hit_r[n > 0] - mean_s[n > 0])
                            * (n[n > 0] / tot))) if tot else float("nan"))
        out["calibration"] = {
            "bin_edges": self._cal_edges.copy(), "bin_n": n.copy(),
            "bin_mean_score": mean_s, "bin_hit_rate": hit_r,
            "ece": ece, "n": tot}
        return out


# ── aggregation helpers ──────────────────────────────────────────────────────

def overall_recall(result: dict, m: int = 8) -> tuple[float, int]:
    """(n-weighted mean recall@m over all labeled cells, total n)."""
    r = result["recall"][m]
    n = r["n"]
    tot = int(n.sum())
    if tot == 0:
        return float("nan"), 0
    mean = np.nansum(np.where(n > 0, r["mean"] * n, 0.0)) / tot
    return float(mean), tot


def per_position_recall(result: dict, m: int = 8) -> list[dict]:
    """Rows of {k, recall, n} — the position-marginal fidelity (the W[k]
    empirical input), n-weighted over layers."""
    r = result["recall"][m]
    rows = []
    for k in range(result["max_gamma"]):
        n = int(r["n"][k].sum())
        v = (float(np.nansum(np.where(r["n"][k] > 0,
                                      r["mean"][k] * r["n"][k], 0.0)) / n)
             if n else float("nan"))
        rows.append({"k": k, "recall": v, "n": n})
    return rows


def wmap_weighted_recall(result: dict, w: np.ndarray, m: int = 8,
                         ) -> tuple[float, float]:
    """Deployment-relevant aggregate: sum(W[k,j] * recall@m[k,j]) /
    sum(W[k,j]) over LABELED cells (n > 0), with W renormalized over the
    observed cells (EPM-4). Per-cell recall is the cell MEAN (each (k, j)
    cell counts by its value W, not by its label count n — deep positions
    are label-scarce by construction, TD-EPM-REJECTED-LABELS, but their
    prefetch value is what W says). Returns (score, coverage) where
    coverage = fraction of total W mass that was observable (n > 0)."""
    r = result["recall"][m]
    mean, n = np.asarray(r["mean"], np.float64), np.asarray(r["n"])
    w = np.asarray(w, np.float64)
    if w.shape != mean.shape:
        raise ValueError(f"wmap shape {w.shape} != recall {mean.shape}")
    obs = n > 0
    tot = float(w.sum())
    if tot <= 0 or not obs.any():
        return float("nan"), 0.0
    score = float((w[obs] * mean[obs]).sum() / w[obs].sum())
    return score, float(w[obs].sum() / tot)


# ── serialization ────────────────────────────────────────────────────────────

def save_result(result: dict, prefix: str | Path, meta: dict | None = None,
                ) -> None:
    """Write <prefix>.npz (heatmap arrays) + <prefix>.json (aggregates).
    See the module docstring for the exact key contract (EPM-3 input)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = {"moe_layers": result["moe_layers"],
                    "n_missing_pred": result["n_missing_pred"],
                    "union_coverage": result["union_coverage"]["mean"],
                    "union_n": result["union_coverage"]["n"]}
    for m, r in result["recall"].items():
        arrays[f"recall{m}"] = r["mean"]
        arrays[f"n{m}"] = r["n"]
    cal = result["calibration"]
    arrays["calib_bin_edges"] = cal["bin_edges"]
    arrays["calib_bin_n"] = cal["bin_n"]
    arrays["calib_bin_mean_score"] = cal["bin_mean_score"]
    arrays["calib_bin_hit_rate"] = cal["bin_hit_rate"]
    np.savez(str(prefix) + ".npz", **arrays)

    summary: dict = {
        "max_gamma": result["max_gamma"],
        "n_layers": int(len(result["moe_layers"])),
        "ece": cal["ece"], "calib_n": cal["n"],
        "per_position_recall8": per_position_recall(result, 8),
        "n_missing_pred_total": int(result["n_missing_pred"].sum()),
    }
    for m in result["recall"]:
        v, n = overall_recall(result, m)
        summary[f"overall_recall{m}"] = v
        summary[f"overall_n{m}"] = n
    un = result["union_coverage"]
    tot = int(un["n"].sum())
    summary["overall_union_coverage"] = (
        float(np.nansum(np.where(un["n"] > 0, un["mean"] * un["n"], 0.0))
              / tot) if tot else float("nan"))
    summary["union_n_total"] = tot
    if meta:
        summary["meta"] = meta
    with open(str(prefix) + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, allow_nan=True)


def load_result_npz(prefix: str | Path) -> dict:
    """Load the npz half back into the result() dict shape (recall/union/
    calibration) — the EPM-3-side reader."""
    z = np.load(str(Path(prefix)) + ".npz")
    result: dict = {"moe_layers": z["moe_layers"],
                    "max_gamma": int(z["recall8"].shape[0]),
                    "n_missing_pred": z["n_missing_pred"], "recall": {}}
    for m in RECALL_MS:
        if f"recall{m}" in z:
            result["recall"][m] = {"mean": z[f"recall{m}"], "n": z[f"n{m}"]}
    result["union_coverage"] = {"mean": z["union_coverage"],
                                "n": z["union_n"]}
    n = z["calib_bin_n"]
    tot = int(n.sum())
    hr, ms_ = z["calib_bin_hit_rate"], z["calib_bin_mean_score"]
    ece = (float(np.sum(np.abs(hr[n > 0] - ms_[n > 0]) * (n[n > 0] / tot)))
           if tot else float("nan"))
    result["calibration"] = {"bin_edges": z["calib_bin_edges"], "bin_n": n,
                             "bin_mean_score": ms_, "bin_hit_rate": hr,
                             "ece": ece, "n": tot}
    return result


def save_heatmap_png(arr: np.ndarray, n: np.ndarray, path: str | Path,
                     title: str, moe_layers: np.ndarray) -> bool:
    """Position × layer heatmap PNG.  Returns False (and writes nothing)
    when matplotlib is unavailable — the harness never requires it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    g, j = arr.shape
    fig, ax = plt.subplots(figsize=(max(6.0, j * 0.14), max(2.5, g * 0.5)))
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0,
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel("MoE layer (absolute id)")
    ax.set_ylabel("block position k")
    step = max(1, j // 16)
    ax.set_xticks(range(0, j, step))
    ax.set_xticklabels([str(int(moe_layers[i])) for i in range(0, j, step)],
                       fontsize=7)
    ax.set_yticks(range(g))
    ax.set_title(f"{title}  (n={int(n.sum())})")
    fig.colorbar(im, ax=ax, label="recall")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def summary_table(results_by_name: dict[str, dict]) -> str:
    """Markdown fidelity table across baseline variants (rows) with overall
    + per-position recall@8, recall@16/32, union coverage, ECE — every
    number annotated with its n (coverage-awareness is part of the report
    contract)."""
    if not results_by_name:
        return "(no results)\n"
    any_res = next(iter(results_by_name.values()))
    g = any_res["max_gamma"]
    hdr = (["baseline", "recall@8 (n)", "recall@16 (n)", "recall@32 (n)",
            "union-cov (n)", "ECE (n)"]
           + [f"k={k} r@8 (n)" for k in range(g)])
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "---|" * len(hdr)]

    def fmt(v: float, n: int) -> str:
        return "—" if n == 0 or not np.isfinite(v) else f"{v:.4f} ({n})"

    for name, res in results_by_name.items():
        row = [name]
        for m in (8, 16, 32):
            if m in res["recall"]:
                v, n = overall_recall(res, m)
                row.append(fmt(v, n))
            else:
                row.append("—")
        un = res["union_coverage"]
        tot = int(un["n"].sum())
        uv = (float(np.nansum(np.where(un["n"] > 0, un["mean"] * un["n"],
                                       0.0)) / tot) if tot else float("nan"))
        row.append(fmt(uv, tot))
        cal = res["calibration"]
        row.append(fmt(cal["ece"], cal["n"]))
        for r in per_position_recall(res, 8):
            row.append(fmt(r["recall"], r["n"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
