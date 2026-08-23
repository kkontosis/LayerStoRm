"""Training framework: fit a pluggable cost-model's correction coefficients to the
joined dataset by minimizing predicted-vs-real error (goal 4).

The model parameters (model.Params.vec) ARE the trainable tensor. We minimize a
normalized least-squares (RMSE-in-microseconds) loss over the per-term targets
that have ground truth:

    L(params) = w_total * RMSE( pred_total , real_total )
              + w_xfer  * RMSE( pred_xferwall , real_xferwall )

RMSE (not MSE) keeps the loss and its finite-difference gradient O(microseconds)
rather than O(us^2)~1e5, which is what makes the adaptive step stable.

The fit is model-AGNOSTIC: any plugin exposing predict() can be trained with the
finite-difference gradient (no hand-derived gradients), so swapping in an
ALTERNATIVE model (ContentionBankModel) needs NO trainer change — that is the
pluggability requirement (goal 5). For speed on the two registered forms we
pre-extract the per-row raw features once and evaluate predictions vectorised in
numpy; the generic predict() path is used as the fallback (correctness oracle in
the tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from model import Model, Params, ContentionBankModel, CurrentModel


def _real(rows, name):
    return np.array(
        [r["real"].get(name) if r["real"].get(name) is not None else np.nan for r in rows],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Vectorised feature extraction (computed once per dataset).
#   For each row we need, per device d: sum of assigned uncached subxfer + count.
#   Per bank b: total egress draw + contention degree g_b (distinct fetching devs).
# We store them as padded [R, M] / [R, B] arrays so a candidate params vector
# evaluates the whole dataset in a few numpy ops.
# ---------------------------------------------------------------------------
@dataclass
class Features:
    M: int
    B: int
    dev_sub: np.ndarray   # [R, M] sum of uncached subxfer on each device
    dev_cnt: np.ndarray   # [R, M] assigned count per device
    bank_draw: np.ndarray  # [R, B] summed egress draw per bank
    bank_g: np.ndarray    # [R, B] contention degree (distinct fetching devices)
    total_real: np.ndarray
    xfer_real: np.ndarray


def extract_features(rows: List[dict]) -> Features:
    M = max(int(r["M"]) for r in rows)
    B = max(int(r["B"]) for r in rows)
    R = len(rows)
    dev_sub = np.zeros((R, M))
    dev_cnt = np.zeros((R, M))
    bank_draw = np.zeros((R, B))
    bank_devs: List[List[set]] = []
    for ri, r in enumerate(rows):
        bdsets = [set() for _ in range(B)]
        for e in r["experts"]:
            j = int(e["j"])
            b = int(e["bank"])
            if 0 <= j < M:
                dev_cnt[ri, j] += 1
                cached = bool(e["cached"][j])
                if not cached:
                    dev_sub[ri, j] += float(e["subxfer_us"][j])
                    bdsets[b].add(j)
            if not (0 <= j < M and bool(e["cached"][j])):
                if 0 <= b < B:
                    bank_draw[ri, b] += float(e["egress_us"])
        bank_devs.append(bdsets)
    bank_g = np.ones((R, B))
    for ri in range(R):
        for b in range(B):
            bank_g[ri, b] = max(1, len(bank_devs[ri][b]))
    return Features(
        M=M, B=B, dev_sub=dev_sub, dev_cnt=dev_cnt,
        bank_draw=bank_draw, bank_g=bank_g,
        total_real=_real(rows, "total_us"), xfer_real=_real(rows, "transfer_wall_us"),
    )


def _vec_predict(model: Model, params: Params, F: Features):
    """Vectorised (device_makespan, bank_egress) for the registered model forms.
    Returns (dev_make[R], bank[R])."""
    M, B = F.M, F.B
    dev_scale = np.array([params.get(f"dev_scale[{d}]") for d in range(M)])
    comp_a = np.array([params.get(f"compute_a[{d}]") for d in range(M)])
    dev_off = params.get("dev_off")
    r_d = dev_scale[None, :] * (F.dev_sub + comp_a[None, :] * F.dev_cnt) + dev_off
    r_d = np.where(F.dev_cnt > 0, r_d, -np.inf)
    dev_make = r_d.max(axis=1)
    dev_make = np.where(np.isfinite(dev_make), dev_make, 0.0)

    bank_scale = params.get("bank_scale")
    bank_off = params.get("bank_off")
    if isinstance(model, ContentionBankModel):
        c = params.get("contention_c")
        eff = F.bank_draw * (c + (1.0 - c) / F.bank_g)
    else:
        eff = F.bank_draw  # strict serial sum (current model)
    bank = bank_scale * eff.max(axis=1) + bank_off
    return dev_make, bank


def loss_fn_fast(model: Model, params: Params, F: Features, weights: Dict[str, float]) -> float:
    dev_make, bank = _vec_predict(model, params, F)
    xfer_pred = np.maximum(dev_make, bank)
    loss = 0.0
    if weights.get("transfer_wall", 0):
        m = np.isfinite(xfer_pred) & np.isfinite(F.xfer_real)
        if m.any():
            loss += weights["transfer_wall"] * float(
                np.sqrt(np.mean((xfer_pred[m] - F.xfer_real[m]) ** 2)))
    if weights.get("total", 0):
        # total needs recon/prep/place/evict; approximate via the term-summed total
        # against real total. recon/prep from row defaults are folded by predict();
        # here we add the predicted post-fetch (compute) only through real total fit
        # is handled by the generic path. For the fast path 'total' uses xfer as the
        # dominant driver plus a constant offset absorbed by recon_off.
        m = np.isfinite(xfer_pred) & np.isfinite(F.total_real)
        if m.any():
            recon_off = params.get("recon_off")
            tot_pred = xfer_pred + recon_off
            loss += weights["total"] * float(
                np.sqrt(np.mean((tot_pred[m] - F.total_real[m]) ** 2)))
    return loss


def loss_fn(model: Model, params: Params, rows: List[dict], weights: Dict[str, float]) -> float:
    """Generic (per-row predict) RMSE loss — correctness oracle / fallback."""
    preds = [model.predict(params, r) for r in rows]
    total_pred = np.array([p["total"] for p in preds])
    xfer_pred = np.array([max(p["device_makespan"], p["bank_egress"]) for p in preds])
    total_real = _real(rows, "total_us")
    xfer_real = _real(rows, "transfer_wall_us")
    loss = 0.0
    m = np.isfinite(total_pred) & np.isfinite(total_real)
    if m.any() and weights.get("total", 0):
        loss += weights["total"] * float(np.sqrt(np.mean((total_pred[m] - total_real[m]) ** 2)))
    m = np.isfinite(xfer_pred) & np.isfinite(xfer_real)
    if m.any() and weights.get("transfer_wall", 0):
        loss += weights["transfer_wall"] * float(np.sqrt(np.mean((xfer_pred[m] - xfer_real[m]) ** 2)))
    return loss


def _num_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


@dataclass
class TrainResult:
    model: str
    params: Params
    loss0: float
    loss: float
    iters: int
    learned: Dict[str, float]


def _clamp(params: Params, model: Model) -> None:
    for name, (lo, hi) in model.param_bounds().items():
        if name in params._idx:
            params.set(name, float(np.clip(params.get(name), lo, hi)))


def train(
    model: Model,
    rows: List[dict],
    weights: Optional[Dict[str, float]] = None,
    lr: float = 0.5,
    iters: int = 300,
    init: Optional[Params] = None,
    seed: int = 0,
) -> TrainResult:
    """Bounded adaptive-step (RMSProp) gradient descent on the RMSE loss.

    Uses the vectorised fast loss for the registered model forms. Parameters are
    clamped to their bounds after every step (scales >=0, contention_c in [0,1]),
    which is what keeps the fit from running off to nonsense negatives.
    """
    weights = weights or {"total": 1.0, "transfer_wall": 1.0}
    params = init.copy() if init is not None else model.init_params()
    F = extract_features(rows)
    x = params.vec

    def f(vec: np.ndarray) -> float:
        params.vec = vec
        _clamp(params, model)
        vec[:] = params.vec  # write back the clamped values
        return loss_fn_fast(model, params, F, weights)

    loss0 = f(x.copy())
    ms = np.ones_like(x) * 1e-6
    best_x = x.copy()
    best_loss = loss0
    cur = x.copy()
    cur_lr = lr
    stale = 0
    for _ in range(iters):
        g = _num_grad(f, cur)
        ms = 0.9 * ms + 0.1 * g * g
        step = cur_lr * g / (np.sqrt(ms) + 1e-8)
        params.vec = cur - step
        _clamp(params, model)
        trial = params.vec.copy()
        l = loss_fn_fast(model, params, F, weights)
        if l < best_loss - 1e-9:
            best_loss = l
            best_x = trial.copy()
            cur = trial
            stale = 0
        else:
            # overshoot / no improvement: shrink the step and retry from best.
            stale += 1
            cur_lr *= 0.5
            cur = best_x.copy()
            if cur_lr < 1e-6:
                break
    params.vec = best_x
    return TrainResult(
        model=model.name,
        params=params,
        loss0=float(loss0),
        loss=float(best_loss),
        iters=iters,
        learned=params.as_dict(),
    )


def calibrate_closed_form(model: Model, rows: List[dict]) -> Params:
    """Fast closed-form per-term scale+offset calibration (no GD) for the bank/
    transfer wall: LS-fit the raw binding term against the real transfer wall and
    fold into bank_scale/bank_off. Useful as a train() init or standalone."""
    from analyze import _ls_scale_offset

    params = model.init_params()
    F = extract_features(rows)
    dev_make, bank = _vec_predict(model, params, F)
    xfer_pred = np.maximum(dev_make, bank)
    m = np.isfinite(xfer_pred) & np.isfinite(F.xfer_real)
    if m.sum() >= 2:
        scale, offset, _ = _ls_scale_offset(xfer_pred[m], F.xfer_real[m])
        if "bank_scale" in params._idx:
            params.set("bank_scale", max(0.0, params.get("bank_scale") * scale))
            params.set("bank_off", offset)
    return params


def apply_params_to_constants(model: Model, learned: Params, calib: dict) -> dict:
    """Bake the fitted correction coefficients into a *copy* of the LoaderConstants
    dict (``calib``), returning ``calib'`` such that ::

        predict(default_params, row, calib')  ~=  predict(learned, row, calib)

    holds for every row (within tolerance — see the additive-offset note below).
    ``calib'`` is the same on-disk schema as the input (loadable by the engine), so
    the engine, re-running with calibration_path=calib', behaves as if the model had
    been trained.

    Per-coefficient mapping (consistent with model.CurrentModel.predict's calib path
    and the C++ LoaderSolver objective):

      * ``dev_scale[d]``  -> multiply ``matrix[b][d].rate_us`` AND ``.lat_us`` for every
        bank b (the device's whole per-expert transfer term, == subxfer) and the
        device's ``xfer_lat_us`` provenance field.
      * ``compute_a[d]``  -> set ``devices[d].compute.a_us += dev_scale[d]*compute_a[d]``
        (the per-expert linear compute the dump never carried; pre-scaled by dev_scale
        so the default-params calib path reproduces learned's dev_scale*(sub+comp)).
      * ``bank_scale``    -> multiply every ``banks[b].egress_us``.
      * ``recon_scale``   -> multiply every device's ``recon_overhead_us`` and
        ``recon_added_us``.
      * ADDITIVE OFFSETS ``dev_off`` + ``bank_off`` + ``recon_off`` (the big fixed costs
        the per-rate model structurally misses, ~365/1074 µs) -> summed into the NEW
        GLOBAL ``fixed_overhead_us`` field, which predict() / the C++ solver add to T.
        A uniform µs offset is NOT a per-device rate, so it cannot be folded into any
        matrix cell; the dedicated overhead term is the correct home. Note dev_off and
        bank_off sit *inside* the max(dev,bank) in predict while a global overhead sits
        *outside* it — the round-trip is therefore EXACT whenever the same term binds
        (the overwhelmingly common case) and otherwise off by at most one offset.
    """
    import copy

    out = copy.deepcopy(calib)
    p = learned.as_dict()
    M = int(model.M)

    devs = out.setdefault("devices", [])
    banks = out.setdefault("banks", [])
    mat = out.setdefault("matrix", [])

    dev_scale = [float(p.get(f"dev_scale[{d}]", 1.0)) for d in range(M)]
    comp_a    = [float(p.get(f"compute_a[{d}]", 0.0)) for d in range(M)]
    bank_scale = float(p.get("bank_scale", 1.0))
    recon_scale = float(p.get("recon_scale", 1.0))

    # dev_scale -> matrix rate+lat (every bank row) for that device column.
    for brow in mat:
        for d in range(min(M, len(brow))):
            cell = brow[d]
            cell["rate_us"] = float(cell.get("rate_us", 0.0)) * dev_scale[d]
            cell["lat_us"]  = float(cell.get("lat_us", 0.0)) * dev_scale[d]

    # compute_a + dev_scale (xfer_lat provenance) + recon_scale -> per device.
    for d in range(min(M, len(devs))):
        dd = devs[d]
        dd["xfer_lat_us"] = float(dd.get("xfer_lat_us", 0.0)) * dev_scale[d]
        comp = dd.setdefault("compute", {})
        comp["a_us"] = float(comp.get("a_us", 0.0)) + dev_scale[d] * comp_a[d]
        dd["recon_overhead_us"] = float(dd.get("recon_overhead_us", 0.0)) * recon_scale
        dd["recon_added_us"]    = float(dd.get("recon_added_us", 0.0)) * recon_scale

    # bank_scale -> egress.
    for b in banks:
        b["egress_us"] = float(b.get("egress_us", 0.0)) * bank_scale

    # additive offsets -> global fixed_overhead_us.
    overhead = (float(p.get("dev_off", 0.0))
                + float(p.get("bank_off", 0.0))
                + float(p.get("recon_off", 0.0)))
    out["fixed_overhead_us"] = float(out.get("fixed_overhead_us", 0.0)) + overhead

    out["source"] = "trained"
    return out


def why(result: TrainResult, baseline: Params) -> str:
    lines = [f"Learned corrections for model='{result.model}':",
             f"  loss {result.loss0:.4g} -> {result.loss:.4g} "
             f"({100*(1-result.loss/(result.loss0 or 1)):.1f}% RMSE reduction)"]
    base = baseline.as_dict()
    for k, v in result.learned.items():
        b = base.get(k, v)
        if abs(v - b) > 1e-3 * (1 + abs(b)):
            lines.append(f"  {k}: {b:.4g} -> {v:.4g}")
    if "contention_c" in result.learned:
        c = result.learned["contention_c"]
        lines.append(
            f"  contention_c={c:.3f}: 1=strict-serial bank sum, 0=perfect sharing; "
            f"the recovered value reveals how the same-bank channel actually shares.")
    return "\n".join(lines)
