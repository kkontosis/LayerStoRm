"""EPM-3 W[k][j] value map + capacity allocation (Phase 29,
spec/MoE-SpeQ_NOTES.md §3 "Position × layer value map").

W[k][j] encodes the PREFETCH VALUE of predicting position k's routing at
MoE layer j.  It is used twice (confirmed directive):
  (a) as the per-cell training-loss weight (train.py), and
  (b) to allocate trainable capacity — the per-position delta-adapter rank
      schedule (allocate_ranks below, consumed by model.py).

Directive defaults (kind="directive"; the EXACT default, documented):
  - k = 0 is MAXIMAL on ALL layers: W[0][j] = k0_value (default 1.0) —
    position 0 is verified in every block.
  - k >= 1 baseline = k0_value * survival[k] * depth_taper[j], floored at
    `floor` (default 0.02: keeps a small gradient signal everywhere; the
    capacity path can still assign rank 0).
      survival[k]   = prod(accept_curve[:k]), survival[0] = 1 — the
                      accept_curve entries are CONDITIONAL per-position
                      acceptance probabilities a_k = P(position k survives
                      | positions < k survived) (DSpark's c_k semantics),
                      composed by cumulative product into the probability
                      that position k is verified at all.  Default curve =
                      the measured DSP-6 acceptance curve
                      (0.43, 0.28, 0.27, 0.21, 0.11, 0.07, 0.09); note it
                      is NOT itself a survival vector (it is not monotone:
                      0.07 -> 0.09) — the cumprod is what makes W's
                      position taper monotone.  Curves shorter than needed
                      are extended with their last entry.
  - k = 1 early boost: W[1][j] for the FIRST k1_early_layers entries of
    moe_layers (default 4 — GLM-5.2 absolute layers 3..6) is raised to at
    least k1_early_value * k0_value (default 0.8), capped at k0_value:
    position 1 is verified in ~43% of blocks and the EARLY layers of the
    next-verified token have no alternative early signal.
  - depth_taper[j]: linear 1.0 -> depth_taper_min (default 0.5) across the
    moe_layers INDEX (applies to k >= 1 only; k = 0 stays flat per the
    directive "k=0 maximal on ALL layers").

Other kinds (EPM-4 grid arms): "uniform" (the W-vs-uniform A/B control)
and "explicit" (a full [G, J] matrix straight from the config).

Capacity mapping W -> ranks (allocate_ranks; documented contract):
  value of the (position k, layer-group g) delta adapter
      v[k, g] = sum_{j in g} W[k][j]
  (a shared adapter costs the same regardless of how many layers it
  serves, so its value is the SUM of the value it serves).  One unit of
  rank costs (in_dim + out_dim) parameters.  Ranks are the largest-
  remainder apportionment of budget_params // (in_dim + out_dim) rank
  units proportional to v, clamped to [min_rank, max_rank] (min_rank
  applies only where v > 0).  Guarantees: total parameters <= budget;
  v[a] > v[b] => rank[a] >= rank[b] (before max_rank saturation the
  ordering is strict-or-equal by construction); deterministic (ties break
  toward lower k, then lower g).
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Measured DSP-6 conditional per-position acceptance curve (a_k = P(pos k
# survives | pos <k survived); spec/MoE-SpeQ_NOTES.md §3 "measured a_1 ≈
# 0.43 ... 0.43 → 0.09 over γ=7").
DEFAULT_ACCEPT_CURVE = (0.43, 0.28, 0.27, 0.21, 0.11, 0.07, 0.09)


def survival_from_curve(accept_curve, max_gamma: int) -> np.ndarray:
    """survival[k] = P(position k verified) = prod(accept_curve[:k]);
    survival[0] = 1.  Curve extended with its last entry when max_gamma
    exceeds len(curve) + 1."""
    curve = [float(c) for c in accept_curve]
    if not curve:
        raise ValueError("accept_curve must be non-empty")
    if any(not 0.0 <= c <= 1.0 for c in curve):
        raise ValueError("accept_curve entries must be in [0, 1]")
    while len(curve) < max_gamma - 1:
        curve.append(curve[-1])
    out = np.ones(max_gamma, np.float64)
    for k in range(1, max_gamma):
        out[k] = out[k - 1] * curve[k - 1]
    return out


@dataclasses.dataclass
class WMapSpec:
    """Serializable W[k][j] specification (round-trips through the
    checkpoint sidecar via to_dict/from_dict)."""

    kind: str = "directive"                # directive | uniform | explicit
    # directive knobs:
    accept_curve: tuple = DEFAULT_ACCEPT_CURVE
    k0_value: float = 1.0
    k1_early_layers: int = 4
    k1_early_value: float = 0.8            # relative to k0_value
    depth_taper_min: float = 0.5
    floor: float = 0.02
    # uniform:
    value: float = 1.0
    # explicit:
    matrix: list | None = None             # [G][J] nested lists

    def __post_init__(self) -> None:
        if self.kind not in ("directive", "uniform", "explicit"):
            raise ValueError(f"unknown wmap kind {self.kind!r}")
        if self.kind == "directive":
            if not 0.0 <= self.floor < self.k0_value:
                raise ValueError("need 0 <= floor < k0_value")
            if not 0.0 < self.depth_taper_min <= 1.0:
                raise ValueError("depth_taper_min must be in (0, 1]")
            if not 0.0 <= self.k1_early_value <= 1.0:
                raise ValueError("k1_early_value is relative to k0_value "
                                 "and must be in [0, 1]")
        if self.kind == "explicit" and self.matrix is None:
            raise ValueError("explicit wmap needs a matrix")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["accept_curve"] = list(self.accept_curve)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WMapSpec":
        d = dict(d)
        if "accept_curve" in d:
            d["accept_curve"] = tuple(float(c) for c in d["accept_curve"])
        return cls(**d)

    # ── matrix construction ──────────────────────────────────────────────

    def build(self, max_gamma: int, moe_layers) -> np.ndarray:
        """W float32 [G, J] per the module-docstring defaults."""
        g = int(max_gamma)
        j = len(moe_layers)
        if g < 1 or j < 1:
            raise ValueError("need max_gamma >= 1 and at least one layer")
        if self.kind == "uniform":
            return np.full((g, j), float(self.value), np.float32)
        if self.kind == "explicit":
            w = np.asarray(self.matrix, np.float32)
            if w.shape != (g, j):
                raise ValueError(f"explicit wmap shape {w.shape} != "
                                 f"({g}, {j})")
            if np.any(w < 0):
                raise ValueError("wmap values must be >= 0")
            return w
        # directive
        surv = survival_from_curve(self.accept_curve, g)
        taper = (np.linspace(1.0, self.depth_taper_min, j) if j > 1
                 else np.ones(1))
        w = np.empty((g, j), np.float64)
        w[0, :] = self.k0_value
        for k in range(1, g):
            w[k] = np.maximum(self.floor,
                              self.k0_value * surv[k] * taper)
        if g > 1 and self.k1_early_layers > 0:
            n = min(int(self.k1_early_layers), j)
            boost = min(self.k0_value,
                        self.k1_early_value * self.k0_value)
            w[1, :n] = np.maximum(w[1, :n], boost)
        return w.astype(np.float32)


def build_wmap(spec: dict | WMapSpec, max_gamma: int,
               moe_layers) -> np.ndarray:
    """Config-dict entrypoint: {"kind": ..., ...} -> W float32 [G, J]."""
    if isinstance(spec, dict):
        spec = WMapSpec.from_dict(spec)
    return spec.build(max_gamma, moe_layers)


# ── capacity allocation ──────────────────────────────────────────────────────

def allocate_ranks(w: np.ndarray, group_of_layer, *, budget_params: int,
                   in_dim: int, out_dim: int, min_rank: int = 0,
                   max_rank: int = 256) -> np.ndarray:
    """Delta-adapter rank schedule r[k, g] from W (module docstring
    contract).  w: [G, J]; group_of_layer: int [J] (layer j's adapter
    group).  Returns int64 [G, n_groups]; total params
    (r.sum() * (in_dim + out_dim)) <= budget_params."""
    w = np.asarray(w, np.float64)
    gol = np.asarray(group_of_layer, np.int64)
    if w.ndim != 2 or len(gol) != w.shape[1]:
        raise ValueError("w must be [G, J] with group_of_layer per layer")
    if np.any(w < 0):
        raise ValueError("wmap values must be >= 0")
    if min_rank < 0 or max_rank < min_rank:
        raise ValueError("need 0 <= min_rank <= max_rank")
    g_pos, j = w.shape
    n_groups = int(gol.max()) + 1 if len(gol) else 0
    unit_cost = int(in_dim) + int(out_dim)
    if unit_cost <= 0:
        raise ValueError("in_dim + out_dim must be positive")

    # v[k, g] = value mass served by the (k, g) delta.
    v = np.zeros((g_pos, n_groups), np.float64)
    for jj in range(j):
        v[:, gol[jj]] += w[:, jj]

    total_units = int(budget_params) // unit_cost
    r = np.zeros((g_pos, n_groups), np.int64)
    active = v > 0.0
    r[active] = min_rank
    if r.sum() > total_units:
        raise ValueError(
            f"budget_params={budget_params} cannot cover min_rank="
            f"{min_rank} for {int(active.sum())} active cells at "
            f"{unit_cost} params/rank")
    remaining = total_units - int(r.sum())
    vsum = float(v.sum())
    if remaining > 0 and vsum > 0.0:
        raw = remaining * v / vsum
        extra = np.floor(raw).astype(np.int64)
        extra = np.minimum(extra, max_rank - r)          # clamp headroom
        r += np.where(active, extra, 0)
        leftover = remaining - int(np.where(active, extra, 0).sum())
        if leftover > 0:
            frac = np.where(active, raw - np.floor(raw), -1.0)
            # deterministic order: remainder desc, then value desc, then
            # (k, g) asc.
            order = sorted(
                ((k, gg) for k in range(g_pos) for gg in range(n_groups)
                 if active[k, gg]),
                key=lambda kg: (-frac[kg], -v[kg], kg))
            while leftover > 0:
                progressed = False
                for kg in order:
                    if leftover == 0:
                        break
                    if r[kg] < max_rank:
                        r[kg] += 1
                        leftover -= 1
                        progressed = True
                if not progressed:
                    break  # everything saturated at max_rank: under-spend
    r = np.minimum(r, max_rank)
    assert int(r.sum()) * unit_cost <= budget_params
    return r
