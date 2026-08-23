"""DSpark STS calibration (DSP-7) — Sequential Temperature Scaling.

The DSP-9 scheduler needs trustworthy MAGNITUDES from the DSP-6 confidence
head, not just rankings: raw neural survival probabilities c_k are
systematically over/under-confident.  STS (paper §3.2.1) fits ONE
temperature scalar PER DRAFT POSITION, left to right, each minimizing the
Expected Calibration Error (ECE) of the CUMULATIVE-product survival
a_k = Π_{i≤k} ĉ_i at that position with all earlier temperatures held
fixed — a 1-D grid search per position (paper: ECE 3–8% → ~1%).

Semantics:
  - Per-position temperature acts on the LOGIT:  ĉ = σ(logit(c) / T),
    T > 0 — strictly monotone in c, so within-position candidate RANKINGS
    are provably unchanged (order-preserving); only magnitudes move.
  - Labels come fully observed from the lossless sequential verification:
    round with `accepted` prefix length ⇒ prefix-survival label
    y_k = 1[k < accepted] for EVERY draft position k (DeepSpec evaluator
    convention: cumprod_pred vs accept_prefix_mask).
  - The fit is OFFLINE, consuming a recorded (c_k, accepted) trace
    (DsparkDraft.confidence_trace, or a DeepSpec offline-eval export);
    the γ fitted temperatures go into `speculation.dspark.sts_temperatures`
    and DsparkDraft.survival_confidences applies them BEFORE the cumprod.
  - DeepSpec ships only the diagnostics (deepspec/eval/dspark/
    confidence_head.py — ECE/AUROC/Brier); the fit routine here is ours.

Raw-c_k caveat (INV-DSPARK-CONF): everything downstream of the head must
treat un-calibrated magnitudes as ranking-only until this fit has run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Probabilities are epsilon-clamped before the logit transform (a raw
# sigmoid never emits exact 0/1, but traces may round-trip through f32
# or hand-written tests).
_EPS = 1e-7


@dataclass(frozen=True)
class ConfidenceRound:
    """One verified draft round: raw c_k per position + the accepted
    prefix length (positions k < accepted survived; k >= accepted did not
    survive AS A PREFIX — fully observed labels, see module docstring)."""
    confidences: tuple[float, ...]
    accepted: int


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def apply_temperature(c: float, temperature: float) -> float:
    """ĉ = σ(logit(c) / T).  T=1 is the identity; T>0 required
    (order-preserving: strictly increasing in c for any fixed T)."""
    if temperature <= 0.0:
        raise ValueError(f"STS temperature must be > 0, got {temperature}")
    return 1.0 / (1.0 + math.exp(-logit(c) / temperature))


def expected_calibration_error(probs: list[float], labels: list[int],
                               num_bins: int = 10) -> float:
    """Standard equal-width binned ECE: Σ_b (n_b/N)·|acc_b − conf_b|."""
    if len(probs) != len(labels):
        raise ValueError("probs/labels length mismatch")
    n = len(probs)
    if n == 0:
        return 0.0
    sums = [0.0] * num_bins   # Σ predicted prob per bin
    hits = [0] * num_bins     # Σ labels per bin
    cnts = [0] * num_bins
    for p, y in zip(probs, labels):
        b = min(int(p * num_bins), num_bins - 1)
        sums[b] += p
        hits[b] += y
        cnts[b] += 1
    ece = 0.0
    for b in range(num_bins):
        if cnts[b] == 0:
            continue
        ece += (cnts[b] / n) * abs(hits[b] / cnts[b] - sums[b] / cnts[b])
    return ece


def _default_grid() -> list[float]:
    """Geometric 1-D search grid: 2^(i/8) for i in [-24, 24] — T from
    1/8 to 8 at ~9% steps, always containing the identity T=1."""
    return [2.0 ** (i / 8.0) for i in range(-24, 25)]


def fit_sts_temperatures(
    rounds: list[ConfidenceRound],
    gamma: int,
    *,
    grid: list[float] | None = None,
    num_bins: int = 10,
) -> list[float]:
    """Left-to-right per-position temperature fit (paper §3.2.1).

    Position k's 1-D grid search minimizes the ECE of the cumulative
    survival a_k = (Π_{i<k} ĉ_i, earlier temperatures FIXED) · ĉ_k(T)
    against the prefix labels y_k = 1[k < accepted].  Order-preserving by
    construction (temperatures only rescale logits).  Ties prefer the T
    closest to 1 (identity — stability).  Positions with no observations
    (no round drafted that deep) keep T=1.  Returns exactly `gamma`
    temperatures for `speculation.dspark.sts_temperatures`.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    search = grid if grid is not None else _default_grid()
    if not search or any(t <= 0.0 for t in search):
        raise ValueError("temperature grid must be non-empty and > 0")

    temps: list[float] = []
    # Per-round running prefix product of CALIBRATED earlier positions
    # (earlier temperatures held fixed — the sequential part of STS).
    prefix: dict[int, float] = {
        r: 1.0 for r in range(len(rounds))
    }
    for k in range(gamma):
        obs: list[tuple[int, float, int]] = []  # (round idx, raw c_k, y_k)
        for ri, rnd in enumerate(rounds):
            if k < len(rnd.confidences):
                obs.append((ri, rnd.confidences[k],
                            1 if k < rnd.accepted else 0))
        if not obs:
            temps.append(1.0)
            continue

        best_t = 1.0
        best_ece = float("inf")
        for t in search:
            probs = [prefix[ri] * apply_temperature(c, t)
                     for ri, c, _ in obs]
            labels = [y for _, _, y in obs]
            ece = expected_calibration_error(probs, labels, num_bins)
            if (ece < best_ece - 1e-12
                    or (abs(ece - best_ece) <= 1e-12
                        and abs(math.log(t)) < abs(math.log(best_t)))):
                best_ece = ece
                best_t = t
        temps.append(best_t)

        # Freeze position k: fold its calibrated ĉ into each round's prefix.
        for ri, c, _ in obs:
            prefix[ri] *= apply_temperature(c, best_t)
    return temps


def cumulative_survival_ece(
    rounds: list[ConfidenceRound],
    temperatures: list[float] | tuple[float, ...] = (),
    *,
    num_bins: int = 10,
) -> float:
    """Diagnostic: pooled ECE of the cumulative survival a_k over ALL
    observed (round, position) pairs, with optional STS temperatures
    applied (empty = raw).  What the fit drives down; mirrors DeepSpec's
    evaluator pooling (cumprod_pred vs prefix labels)."""
    probs: list[float] = []
    labels: list[int] = []
    for rnd in rounds:
        a = 1.0
        for k, c in enumerate(rnd.confidences):
            t = (temperatures[k]
                 if k < len(temperatures) else 1.0)
            a *= apply_temperature(c, t) if t != 1.0 else min(max(c, 0.0),
                                                              1.0)
            probs.append(a)
            labels.append(1 if k < rnd.accepted else 0)
    return expected_calibration_error(probs, labels, num_bins)


__all__ = [
    "ConfidenceRound",
    "apply_temperature",
    "expected_calibration_error",
    "fit_sts_temperatures",
    "cumulative_survival_ece",
    "logit",
]
