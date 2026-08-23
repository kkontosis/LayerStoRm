"""Pluggable cost-model plugins for the I8 GPU-loader x-ray (spec/GPU_LOADER_MODEL.md).

The C++ ``LoaderSolver`` evaluates one *fixed* model (§3):

    T = prep + max(device_makespan, bank_egress) + recon + place + evict

The dump (LS_LOADER_SHADOW_DUMP) records, per (cmd_seq, layer): the per-expert
inputs (bank, cache mask, subxfer[j], egress) AND the solver's chosen assignment
j[.] plus every predicted sub-term. The x-ray re-derives the sub-terms from the
*raw inputs* through a **pluggable** ``predict(params, row)`` so that:

  * the current model can be re-evaluated under *trainable* correction coefficients
    (scale + offset, per-term and per-device / per-bank), and
  * an ALTERNATIVE model form (e.g. the contention-aware bank term
    ``bank x (c + (1-c)/g)``) can be dropped in and fit the SAME way, enabling
    apples-to-apples model comparison (goal 5).

A plugin is a ``Model``:  ``predict(params: Params, row: dict) -> dict[str, float]``
returning the per-term microseconds {prep, device_makespan, bank_egress, recon,
place, evict, total}. ``params`` is a flat float vector with a named layout
(``Params``) so the trainer can treat it as a tensor and the analyzer can read
named coefficients back.

Each plugin owns:
  * ``param_spec()``  -> ordered list of (name, default) coefficient slots
  * ``predict(params, row)`` -> per-term dict (total = prep + max(dev,bank) + recon + place + evict)

The "inputs" for a row come from the joined dataset (joiner.py): the per-expert
arrays + the solver assignment j[.]. We recompute the makespan/egress from those
raw arrays under the candidate params so corrections actually bite on the physics,
not just on the already-reduced sub-terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# Canonical term names (must match SolveResult sub-terms + 'total').
TERMS = ["prep", "device_makespan", "bank_egress", "recon", "place", "evict", "total"]


@dataclass
class Params:
    """Flat, named coefficient vector. ``vec`` is the trainable tensor; ``names``
    maps slot -> name. ``get(name)`` / ``set(name)`` read/write by name."""

    names: List[str]
    vec: np.ndarray  # shape [P], float64

    def __post_init__(self) -> None:
        assert len(self.names) == self.vec.shape[0], "names/vec length mismatch"
        self._idx = {n: i for i, n in enumerate(self.names)}

    def get(self, name: str) -> float:
        return float(self.vec[self._idx[name]])

    def set(self, name: str, value: float) -> None:
        self.vec[self._idx[name]] = value

    def as_dict(self) -> Dict[str, float]:
        return {n: float(self.vec[i]) for n, i in self._idx.items()}

    def copy(self) -> "Params":
        return Params(list(self.names), self.vec.copy())


class Model:
    """Base class for a pluggable cost-model form."""

    name = "base"

    def param_spec(self) -> List[Tuple[str, float]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """(lo, hi) clamp per coefficient name. Scales >=0; contention_c in [0,1];
        offsets unconstrained. Names absent from the dict are unbounded."""
        bounds: Dict[str, Tuple[float, float]] = {}
        for name, _ in self.param_spec():
            if name.startswith("dev_scale") or name.endswith("_scale"):
                bounds[name] = (0.0, 50.0)
            elif name.startswith("compute_a"):
                bounds[name] = (0.0, 1e5)
            elif name == "contention_c":
                bounds[name] = (0.0, 1.0)
        return bounds

    def init_params(self) -> Params:
        spec = self.param_spec()
        return Params([n for n, _ in spec], np.array([d for _, d in spec], dtype=np.float64))

    def predict(self, params: Params, row: dict) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Row-input extraction helpers (shared by all plugins).
# A joined row carries, per the dump schema:
#   row['M'], row['B'], row['N']
#   row['experts']  : list of {bank, j, egress_us, cached[M], subxfer_us[M]}
#   row['dev_counts'] (solver per-device counts; recomputed here from j to stay
#                      robust to coarsened dumps)
# ---------------------------------------------------------------------------
def _device_groups(row: dict):
    """Return (M, groups) where groups[d] = list of expert dicts assigned to d."""
    M = int(row["M"])
    groups: List[List[dict]] = [[] for _ in range(M)]
    for e in row["experts"]:
        j = int(e["j"])
        if 0 <= j < M:
            groups[j].append(e)
    return M, groups


def _bank_groups(row: dict):
    """Return (B, banks) where banks[b] = list of (expert, fetched?) drawing bank b."""
    B = int(row["B"])
    banks: List[List[Tuple[dict, bool]]] = [[] for _ in range(B)]
    for e in row["experts"]:
        b = int(e["bank"])
        j = int(e["j"])
        cached_on_j = (0 <= j < int(row["M"])) and bool(e["cached"][j])
        fetched = not cached_on_j
        if 0 <= b < B:
            banks[b].append((e, fetched))
    return B, banks


# ---------------------------------------------------------------------------
# Calibration-sourced raw terms (apply <-> predict round-trip support).
#
# When ``predict`` is given a ``calib`` LoaderConstants dict it recomputes the raw
# physics from the *constants* (matrix rate+lat, bank egress, recon_*, the global
# fixed_overhead_us) using only the row's STRUCTURE (bank, j, cached). This is the
# inverse of ``trainer.apply_params_to_constants``: baking the fitted corrections
# into calib' and predicting at DEFAULT params reproduces the prediction at the
# LEARNED params on the baseline calib (within tol — see apply_params_to_constants
# for how the additive offsets fold into the global fixed_overhead_us).
#
# The mapping mirrors the C++ LoaderSolver objective (loader_solver.cpp
# objective_from_sums): subxfer_d = matrix[bank][d].rate_us + lat_us per uncached
# assigned expert; compute share = devices[d].compute.a_us * count (the per-expert
# linear curve term — the trainer's compute_a[d]); egress draw = banks[b].egress_us
# per fetched expert; recon = devices[d].recon_overhead_us(max) + recon_added_us(sum).
# ---------------------------------------------------------------------------
def calib_subxfer_us(calib: dict, bank: int, dev: int) -> float:
    """Raw subxfer (rate+lat) for one expert sourced from bank b onto device j,
    read from the LoaderConstants matrix (mirrors dispatch_moe.cpp dump)."""
    mat = calib.get("matrix") or []
    if 0 <= bank < len(mat) and 0 <= dev < len(mat[bank]):
        cell = mat[bank][dev]
        return float(cell.get("rate_us", 0.0)) + float(cell.get("lat_us", 0.0))
    return 0.0


def calib_compute_a_us(calib: dict, dev: int) -> float:
    devs = calib.get("devices") or []
    if 0 <= dev < len(devs):
        return float((devs[dev].get("compute") or {}).get("a_us", 0.0))
    return 0.0


def calib_egress_us(calib: dict, bank: int) -> float:
    banks = calib.get("banks") or []
    if 0 <= bank < len(banks):
        return float(banks[bank].get("egress_us", 0.0))
    return 0.0


def calib_recon_us(calib: dict, devs_present: Sequence[int]) -> float:
    """recon = max overhead + sum added over participating devices (objective_from_sums)."""
    devices = calib.get("devices") or []
    ov = 0.0
    ad = 0.0
    any_dev = False
    for d in devs_present:
        if 0 <= d < len(devices):
            ov = max(ov, float(devices[d].get("recon_overhead_us", 0.0)))
            ad += float(devices[d].get("recon_added_us", 0.0))
            any_dev = True
    return (ov + ad) if any_dev else 0.0


def calib_fixed_overhead_us(calib: dict) -> float:
    return float(calib.get("fixed_overhead_us", 0.0))


# ---------------------------------------------------------------------------
# Plugin 1: the CURRENT model (spec §3) with per-term scale+offset corrections
#           and optional per-device makespan scale.
# ---------------------------------------------------------------------------
class CurrentModel(Model):
    """Faithful re-implementation of the C++ LoaderSolver objective, parameterised
    by correction coefficients the trainer fits:

      device_makespan = dev_scale[d*]*(sum subxfer_assigned_d* + compute_d*) + dev_off
        (we recompute the *raw* roofline from the dump's subxfer arrays; compute is
         not in the dump per-expert, so a per-device linear `compute_a` * count is
         added as a trainable term — this is what the GPU run will pin down.)
      bank_egress     = bank_scale * max_b sum_{fetched i in b} egress_i  + bank_off
      recon           = recon_scale * recon_raw + recon_off  (recon_raw from dump)
      total           = prep + max(dev, bank) + recon + place + evict   (all post-correction)

    The makespan recompute is the load-bearing part: it lets a *global* scale and a
    *per-device* scale be fit, exposing systematic device bias the GPU x-ray will reveal.
    """

    name = "current"

    def __init__(self, M: int):
        self.M = M

    def param_spec(self) -> List[Tuple[str, float]]:
        spec: List[Tuple[str, float]] = []
        # per-device makespan scale (default 1) + a per-device linear compute coef (us/expert)
        for d in range(self.M):
            spec.append((f"dev_scale[{d}]", 1.0))
            spec.append((f"compute_a[{d}]", 0.0))
        spec.append(("dev_off", 0.0))
        spec.append(("bank_scale", 1.0))
        spec.append(("bank_off", 0.0))
        spec.append(("recon_scale", 1.0))
        spec.append(("recon_off", 0.0))
        return spec

    def predict(self, params: Params, row: dict, calib: dict = None) -> Dict[str, float]:
        M, groups = _device_groups(row)
        dev_off = params.get("dev_off")
        # Raw-term source: when a ``calib`` LoaderConstants dict is supplied, recompute
        # subxfer/egress/recon/compute FROM the constants (using only row structure),
        # so apply_params_to_constants -> predict(default, calib') round-trips against
        # predict(learned, baseline). Otherwise read the row's pre-baked dump values.
        devs_present = [d for d in range(M) if groups[d]]
        # per-device roofline (barrier mode): serial subxfer on assigned device + compute.
        dev_make = 0.0
        for d in range(M):
            grp = groups[d]
            if not grp:
                continue
            sub = 0.0
            comp_a = params.get(f"compute_a[{d}]")
            for e in grp:
                j = int(e["j"])
                if 0 <= j < M and not bool(e["cached"][j]):
                    if calib is not None:
                        sub += calib_subxfer_us(calib, int(e["bank"]), j)
                    else:
                        sub += float(e["subxfer_us"][j])
            if calib is not None:
                comp_a += calib_compute_a_us(calib, d)
            comp = comp_a * len(grp)
            r_d = params.get(f"dev_scale[{d}]") * (sub + comp) + dev_off
            dev_make = max(dev_make, r_d)

        B, banks = _bank_groups(row)
        raw_bank = 0.0
        for b in range(B):
            if calib is not None:
                s = sum(calib_egress_us(calib, b) for (e, fetched) in banks[b] if fetched)
            else:
                s = sum(float(e["egress_us"]) for (e, fetched) in banks[b] if fetched)
            raw_bank = max(raw_bank, s)
        bank = params.get("bank_scale") * raw_bank + params.get("bank_off")

        prep = float(row.get("prep_us", 0.0))
        recon_raw = (calib_recon_us(calib, devs_present) if calib is not None
                     else float(row.get("recon_us", 0.0)))
        recon = params.get("recon_scale") * recon_raw + params.get("recon_off")
        place = float(row.get("place_us", 0.0))
        evict = float(row.get("evict_us", 0.0))
        overhead = calib_fixed_overhead_us(calib) if calib is not None else 0.0
        total = prep + max(dev_make, bank) + recon + place + evict + overhead
        return {
            "prep": prep,
            "device_makespan": dev_make,
            "bank_egress": bank,
            "recon": recon,
            "place": place,
            "evict": evict,
            "total": total,
        }


# ---------------------------------------------------------------------------
# Plugin 2: contention-aware bank term (the §3.3 / §4.1 OPEN QUESTION).
#   bank_egress = max_b  solo_b * ( c + (1-c)/g_b ) * raw_sum_b_per_expert_avg ...
#
# Concretely we model the per-bank drain as:
#   draw_b   = sum of solo egress over the n_b fetched experts on bank b
#   g_b      = number of DISTINCT devices pulling bank b simultaneously (contention degree)
#   eff_b    = draw_b * ( c + (1 - c) / g_b )
# where c in [0,1] is the trainable "shared-channel" coefficient:
#   c = 1  -> strict serial sum (current model);
#   c = 0  -> perfect parallel sharing across the g_b devices (draw / g_b);
#   intermediate c -> partial contention.
# This is exactly the law the GPU same-bank two-device configs (I8b) calibrate.
# ---------------------------------------------------------------------------
class ContentionBankModel(CurrentModel):
    name = "contention_bank"

    def param_spec(self) -> List[Tuple[str, float]]:
        spec = super().param_spec()
        spec.append(("contention_c", 1.0))  # 1.0 == strict serial == current model
        return spec

    def predict(self, params: Params, row: dict, calib: dict = None) -> Dict[str, float]:
        # reuse the current model's device/recon/prep terms, override bank term.
        base = super().predict(params, row, calib)
        M = int(row["M"])
        B, banks = _bank_groups(row)
        c = params.get("contention_c")
        raw_bank = 0.0
        for b in range(B):
            fetched = [(e, fch) for (e, fch) in banks[b] if fch]
            if not fetched:
                continue
            if calib is not None:
                draw = sum(calib_egress_us(calib, b) for (e, _) in fetched)
            else:
                draw = sum(float(e["egress_us"]) for (e, _) in fetched)
            # distinct target devices among the fetched experts on this bank = contention degree.
            devs = {int(e["j"]) for (e, _) in fetched if 0 <= int(e["j"]) < M}
            g = max(1, len(devs))
            eff = draw * (c + (1.0 - c) / g)
            raw_bank = max(raw_bank, eff)
        bank = params.get("bank_scale") * raw_bank + params.get("bank_off")
        base["bank_egress"] = bank
        overhead = calib_fixed_overhead_us(calib) if calib is not None else 0.0
        base["total"] = (
            base["prep"]
            + max(base["device_makespan"], bank)
            + base["recon"]
            + base["place"]
            + base["evict"]
            + overhead
        )
        return base


REGISTRY: Dict[str, Callable[[int], Model]] = {
    "current": CurrentModel,
    "contention_bank": ContentionBankModel,
}


def make_model(name: str, M: int) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; have {sorted(REGISTRY)}")
    return REGISTRY[name](M)
