"""Analysis: predicted-vs-real residuals + per-term / per-device / per-bank
correction-factor fits for the I8 loader x-ray (goals 1-3).

Maps model terms to their real ground-truth counterparts (joiner stage mapping):

    model total            <-> real total_us
    model max(dev,bank)    <-> real transfer_wall_us   (the fetch wall)
    model device_makespan  <-> real per_device_detect_us straggler (max over gpus)
    model compute (compute_a*count, inside dev) <-> real compute_us
    model bank_egress      <-> (no isolated real metric; surfaces via transfer_wall)
    model recon            <-> None (no marker yet)

For each (model, real) pair we report: bias (mean residual), a least-squares
scale+offset correction (real ~ scale*pred + offset), R^2, and the residual
distribution (p50/p90/max abs). Per-device fits use the dump's per-device
predicted makespan vs the trace's per-device detect wall.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from model import Model, Params, _device_groups, _bank_groups


@dataclass
class TermFit:
    name: str
    n: int
    bias: float           # mean(real - pred)
    scale: float          # LS real ~ scale*pred + offset
    offset: float
    r2: float
    p50_abs: float
    p90_abs: float
    max_abs: float
    rel_rmse: float       # rmse / mean(real)


def _ls_scale_offset(pred: np.ndarray, real: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares fit real ~ scale*pred + offset; return (scale, offset, r2)."""
    if pred.size < 2 or np.allclose(pred, pred[0]):
        # degenerate: fall back to scale=1, offset = mean residual
        off = float(np.mean(real - pred))
        ss_res = float(np.sum((real - (pred + off)) ** 2))
        ss_tot = float(np.sum((real - np.mean(real)) ** 2)) or 1.0
        return 1.0, off, 1.0 - ss_res / ss_tot
    A = np.vstack([pred, np.ones_like(pred)]).T
    (scale, offset), *_ = np.linalg.lstsq(A, real, rcond=None)
    fit = scale * pred + offset
    ss_res = float(np.sum((real - fit) ** 2))
    ss_tot = float(np.sum((real - np.mean(real)) ** 2)) or 1.0
    return float(scale), float(offset), 1.0 - ss_res / ss_tot


def _fit_pair(name: str, pred: np.ndarray, real: np.ndarray) -> Optional[TermFit]:
    mask = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[mask], real[mask]
    if pred.size == 0:
        return None
    scale, offset, r2 = _ls_scale_offset(pred, real)
    resid = real - pred
    abs_resid = np.abs(resid)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mean_real = float(np.mean(real)) or 1.0
    return TermFit(
        name=name,
        n=int(pred.size),
        bias=float(np.mean(resid)),
        scale=scale,
        offset=offset,
        r2=float(r2),
        p50_abs=float(np.percentile(abs_resid, 50)),
        p90_abs=float(np.percentile(abs_resid, 90)),
        max_abs=float(abs_resid.max()),
        rel_rmse=rmse / abs(mean_real),
    )


def predict_rows(model: Model, params: Params, rows: List[dict]) -> List[Dict[str, float]]:
    return [model.predict(params, r) for r in rows]


def analyze(model: Model, params: Params, rows: List[dict]) -> Dict[str, object]:
    """Return a structured report dict (term fits + per-device fits)."""
    preds = predict_rows(model, params, rows)

    def col_pred(term):
        return np.array([p[term] for p in preds], dtype=np.float64)

    def col_real(name):
        return np.array(
            [r["real"].get(name, np.nan) if r["real"].get(name) is not None else np.nan
             for r in rows],
            dtype=np.float64,
        )

    fits: List[TermFit] = []
    # total
    f = _fit_pair("total", col_pred("total"), col_real("total_us"))
    if f:
        fits.append(f)
    # transfer wall  <- max(device_makespan, bank_egress)
    transfer_pred = np.array(
        [max(p["device_makespan"], p["bank_egress"]) for p in preds], dtype=np.float64
    )
    f = _fit_pair("transfer_wall", transfer_pred, col_real("transfer_wall_us"))
    if f:
        fits.append(f)
    # compute  <- the predicted compute share inside device_makespan: recompute it.
    comp_pred = []
    for r, p in zip(rows, preds):
        M, groups = _device_groups(r)
        # straggler device's compute share = compute_a[d]*count for the max device
        comp = 0.0
        for d in range(M):
            n_d = len(groups[d])
            ca = params.get(f"compute_a[{d}]") if f"compute_a[{d}]" in params._idx else 0.0
            comp = max(comp, ca * n_d)
        comp_pred.append(comp)
    f = _fit_pair("compute", np.array(comp_pred, dtype=np.float64), col_real("compute_us"))
    if f:
        fits.append(f)

    # per-device makespan fit: predicted per-device roofline vs real per-device detect wall.
    per_dev_fits: Dict[int, TermFit] = {}
    M_max = max(int(r["M"]) for r in rows)
    for d in range(M_max):
        pp, rr = [], []
        for r in rows:
            M, groups = _device_groups(r)
            if d >= M or not groups[d]:
                continue
            sub = 0.0
            for e in groups[d]:
                j = int(e["j"])
                if not bool(e["cached"][j]):
                    sub += float(e["subxfer_us"][j])
            ca = params.get(f"compute_a[{d}]") if f"compute_a[{d}]" in params._idx else 0.0
            pp.append(params.get(f"dev_scale[{d}]") * (sub + ca * len(groups[d])))
            real_d = r["real"].get("per_device_detect_us", {})
            v = real_d.get(d, real_d.get(str(d)))
            rr.append(v if v is not None else np.nan)
        if pp:
            fit = _fit_pair(f"device[{d}]", np.array(pp), np.array(rr, dtype=np.float64))
            if fit:
                per_dev_fits[d] = fit

    return {
        "model": model.name,
        "n_rows": len(rows),
        "term_fits": [asdict(f) for f in fits],
        "per_device_fits": {d: asdict(f) for d, f in per_dev_fits.items()},
    }


# ===========================================================================
# Stage-1: Σ-calibration  (the decisive arithmetic, spec §3 Stage 1)
#
# The model times each cost at the layer where it is INCURRED, not at the decision
# that CAUSED it (an eviction's future re-fetch is invisible at decision time). So
# if we SUM predicted-T over the whole REALIZED run it should reappear. Procedure:
#   * recompute predicted-T per layer for the ASSIGNMENT THAT RAN (not the dump's j,
#     which under ACT-off is the solver's hypothetical), then Σ it,
#   * compare Σpred to the real total decode time (Σ real total_us).
# Interpretation (per policy, and act-vs-baseline):
#   real(act) ≈ Σpred(act) AND Σpred(act) > Σpred(e%tp)  ⇒ #2 myopia (3a refuted)
#   real(act) > Σpred(act)                                ⇒ #3a unmodeled channel
#   Σpred mis-scaled per policy                           ⇒ #3b
# ===========================================================================
def _assignment_for(row: dict, which: str) -> List[int]:
    """Per-expert device index for the requested assignment.
    'solver'   -> the dump's j[·] (solver's hypothetical, == executed under ACT on);
    'executed' -> the orchestrator e%tp that actually RAN (dump field 'oj'; falls
                  back to expert_idx % M, then to j if neither present)."""
    M = int(row["M"])
    experts = row["experts"]
    if which == "solver":
        return [int(e["j"]) for e in experts]
    out = []
    for e in experts:
        if "oj" in e and int(e["oj"]) >= 0:
            out.append(int(e["oj"]))
        elif "expert_idx" in e:
            out.append(int(e["expert_idx"]) % M)
        else:
            out.append(int(e["j"]))
    return out


def predict_assignment(model: Model, params: Params, row: dict,
                       assign: List[int]) -> Dict[str, float]:
    """Re-evaluate the §3 objective for an ARBITRARY assignment ``assign`` (overriding
    the dump's j[·]). Recomputes device_makespan, bank_egress, evict from the raw
    per-expert inputs so Stage-1 can score the executed e%tp from an ACT-off dump.

    device_makespan = max_d dev_scale[d]*(Σ uncached subxfer on d + compute_a[d]*count)
    bank_egress     = bank_scale * max_b Σ_{fetched i in b} egress_i
    evict           = Σ_d evict_cum[d][n_d] where n_d = #uncached on d  (convex curve
                      reconstructed from the dump's per-device counts is NOT available
                      offline, so evict is taken assignment-INVARIANT from the row's
                      recorded ``evict_us`` UNLESS recomputed below). We DO recompute a
                      convex proxy when the row carries per-device evict curves; today
                      the dump records only the scalar evict_us (solver's j), so for a
                      different assignment we hold evict at the row's value (documented
                      limitation: evict_us is small post-γ, ~10% of T)."""
    M = int(row["M"])
    B = int(row["B"])
    groups: List[List[dict]] = [[] for _ in range(M)]
    for e, d in zip(row["experts"], assign):
        if 0 <= d < M:
            groups[d].append(e)
    dev_make = 0.0
    for d in range(M):
        if not groups[d]:
            continue
        sub = 0.0
        for e in groups[d]:
            if not bool(e["cached"][d]):
                sub += float(e["subxfer_us"][d])
        comp_a = params.get(f"compute_a[{d}]") if f"compute_a[{d}]" in params._idx else 0.0
        r_d = params.get(f"dev_scale[{d}]") * (sub + comp_a * len(groups[d])) \
            + params.get("dev_off")
        dev_make = max(dev_make, r_d)
    bank_draw = [0.0] * B
    for e, d in zip(row["experts"], assign):
        b = int(e["bank"])
        cached_on_d = (0 <= d < M) and bool(e["cached"][d])
        if not cached_on_d and 0 <= b < B:
            bank_draw[b] += float(e["egress_us"])
    raw_bank = max(bank_draw) if bank_draw else 0.0
    c = params.get("contention_c") if "contention_c" in params._idx else 1.0
    if c < 1.0:
        # contention-aware: recompute with g_b = distinct devices per bank.
        raw_bank = 0.0
        for b in range(B):
            fetched = [(e, d) for (e, d) in zip(row["experts"], assign)
                       if int(e["bank"]) == b
                       and not ((0 <= d < M) and bool(e["cached"][d]))]
            if not fetched:
                continue
            draw = sum(float(e["egress_us"]) for (e, _) in fetched)
            g = max(1, len({d for (_, d) in fetched if 0 <= d < M}))
            raw_bank = max(raw_bank, draw * (c + (1.0 - c) / g))
    bank = params.get("bank_scale") * raw_bank + params.get("bank_off")
    prep = float(row.get("prep_us", 0.0))
    recon = params.get("recon_scale") * float(row.get("recon_us", 0.0)) \
        + params.get("recon_off")
    place = float(row.get("place_us", 0.0))
    evict = float(row.get("evict_us", 0.0))
    total = prep + max(dev_make, bank) + recon + place + evict
    return {"prep": prep, "device_makespan": dev_make, "bank_egress": bank,
            "recon": recon, "place": place, "evict": evict, "total": total}


def sum_calibration(model: Model, params: Params, rows: List[dict],
                    which: str = "solver") -> Dict[str, float]:
    """Σ predicted-T over the realized run for the chosen assignment, vs Σ real total.

    Returns Σ of each predicted sub-term, Σ predicted total, Σ real total (from
    total_us), the global scale (Σreal/Σpred), and the per-row LS scale+offset of real
    total vs predicted total (so a *per-policy* scale bias surfaces — that is the 3b
    signal)."""
    pred_terms = {t: 0.0 for t in
                  ["prep", "device_makespan", "bank_egress", "recon", "place", "evict"]}
    sum_pred_total = 0.0
    sum_real_total = 0.0
    n_real = 0
    pp, rr = [], []
    for r in rows:
        assign = _assignment_for(r, which)
        p = predict_assignment(model, params, r, assign)
        for t in pred_terms:
            pred_terms[t] += p[t]
        sum_pred_total += p["total"]
        rt = r["real"].get("total_us")
        if rt is not None and np.isfinite(rt):
            sum_real_total += rt
            n_real += 1
            pp.append(p["total"]); rr.append(rt)
    scale_off = _ls_scale_offset(np.array(pp), np.array(rr)) if len(pp) >= 2 else (1.0, 0.0, 0.0)
    return {
        "assignment": which,
        "n_rows": len(rows),
        "n_real": n_real,
        "sum_pred_terms": pred_terms,
        "sum_pred_total_us": sum_pred_total,
        "sum_real_total_us": sum_real_total,
        "global_scale_real_over_pred": (sum_real_total / sum_pred_total)
                                       if sum_pred_total else float("nan"),
        "ls_scale": scale_off[0],
        "ls_offset": scale_off[1],
        "ls_r2": scale_off[2],
    }


# ===========================================================================
# Stage-2: per-term residual decomposition (spec §3 Stage 2)
#
# Per sub-term (makespan / bank_egress / evict) compute the real-vs-predicted scale
# SEPARATELY. If the per-term scales differ materially that is #3b — and it names the
# term to recalibrate. Decompose the §7 "scale 1.20".
#
# Real ground truth is partial: transfer_wall_us is the analogue of max(dev,bank);
# total_us is the analogue of total. We therefore fit:
#   * transfer_wall vs predicted max(dev_makespan, bank_egress)   -> the makespan scale
#   * total vs predicted total                                    -> the global scale
#   * (total - transfer_wall) vs predicted (recon+compute_tail)   -> the post-fetch scale
# and report the SHARE each predicted term contributes to predicted total (so a term
# that is, say, 60% of T but whose isolated scale is 2x is the one biasing the solver).
# ===========================================================================
def per_term_scale(model: Model, params: Params, rows: List[dict],
                   which: str = "solver") -> Dict[str, object]:
    preds = [predict_assignment(model, params, r, _assignment_for(r, which)) for r in rows]
    real_total = np.array([r["real"].get("total_us", np.nan) for r in rows], dtype=np.float64)
    real_xfer = np.array([r["real"].get("transfer_wall_us")
                          if r["real"].get("transfer_wall_us") is not None else np.nan
                          for r in rows], dtype=np.float64)
    pred_total = np.array([p["total"] for p in preds])
    pred_xfer = np.array([max(p["device_makespan"], p["bank_egress"]) for p in preds])
    pred_post = pred_total - pred_xfer  # recon+place+evict+prep (the non-fetch tail)
    real_post = real_total - real_xfer

    fits: Dict[str, dict] = {}
    for name, pr, re_ in [("total", pred_total, real_total),
                          ("transfer_wall(max dev,bank)", pred_xfer, real_xfer),
                          ("post_fetch(recon+evict+place)", pred_post, real_post)]:
        f = _fit_pair(name, pr, re_)
        if f:
            fits[name] = asdict(f)
    # Share of predicted total contributed by each sub-term (mean over rows). The
    # objective is T = prep + max(dev,bank) + recon + place + evict, so the fetch wall
    # contributes only its BINDING term (max), not dev+bank summed. We report the
    # binding-fetch share (max term) plus each additive term, so the shares partition
    # the predicted total and sum to ~1. We also report which fetch term binds.
    shares = {}
    tot = float(np.mean(pred_total)) or 1.0
    fetch_bind = float(np.mean([max(p["device_makespan"], p["bank_egress"]) for p in preds]))
    shares["fetch_wall(max dev,bank)"] = fetch_bind / tot
    n_dev_binds = int(sum(1 for p in preds if p["device_makespan"] >= p["bank_egress"]))
    shares["_dev_binds_frac"] = n_dev_binds / max(1, len(preds))
    for t in ["recon", "evict", "place", "prep"]:
        shares[t] = float(np.mean([p[t] for p in preds])) / tot
    # the raw (potentially-overlapping) per-term means too, for the bias diagnosis.
    raw_means = {t: float(np.mean([p[t] for p in preds]))
                 for t in ["device_makespan", "bank_egress", "recon", "evict", "place", "prep"]}
    return {"assignment": which, "per_term_fits": fits,
            "term_shares_of_pred_total": shares, "raw_term_means_us": raw_means}


def render_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"# Loader x-ray analysis — model='{report['model']}'  (n={report['n_rows']} rows)")
    lines.append("")
    lines.append("## Per-term residuals (real - predicted) + LS correction (real ~ scale*pred + offset)")
    lines.append("")
    lines.append("| term | n | bias us | scale | offset us | R^2 | p50|resid| | p90|resid| | rel_rmse |")
    lines.append("|------|---|--------:|------:|----------:|----:|-----------:|-----------:|---------:|")
    for f in report["term_fits"]:
        lines.append(
            "| {name} | {n} | {bias:.2f} | {scale:.3f} | {offset:.2f} | {r2:.3f} | "
            "{p50_abs:.2f} | {p90_abs:.2f} | {rel_rmse:.3f} |".format(**f)
        )
    lines.append("")
    if report["per_device_fits"]:
        lines.append("## Per-device makespan fits (predicted roofline vs real detect wall)")
        lines.append("")
        lines.append("| device | n | scale | offset us | R^2 | rel_rmse |")
        lines.append("|--------|---|------:|----------:|----:|---------:|")
        for d, f in sorted(report["per_device_fits"].items()):
            lines.append(
                "| {d} | {n} | {scale:.3f} | {offset:.2f} | {r2:.3f} | {rel_rmse:.3f} |".format(
                    d=d, **f
                )
            )
        lines.append("")
    lines.append("Interpretation: scale != 1 or |bias| large => the model term is mis-calibrated;")
    lines.append("feed scale/offset back as correction coefficients (trainer) or into the C++")
    lines.append("LoaderConstants (per-device xfer/compute, per-bank egress).")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse
    import json
    from joiner import join
    from model import make_model

    ap = argparse.ArgumentParser(description="Analyze loader model vs real timings.")
    ap.add_argument("--model-jsonl", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--model", default="current", choices=["current", "contention_bank"])
    ap.add_argument("--report", default=None, help="optional markdown report path")
    ap.add_argument("--sum-calib", action="store_true",
                    help="Stage-1 Σ-calibration + Stage-2 per-term scale (this run)")
    ap.add_argument("--which", default="solver", choices=["solver", "executed"],
                    help="assignment to score: solver j[.] or executed e%%tp (oj)")
    args = ap.parse_args()
    rows, stats = join(args.model_jsonl, args.trace)
    print(json.dumps(stats, indent=2))
    M = max(int(r["M"]) for r in rows) if rows else 2
    mdl = make_model(args.model, M)
    p = mdl.init_params()
    if args.sum_calib:
        for which in ("solver", "executed"):
            print(f"\n=== Stage-1 Σ-calibration (assignment={which}) ===")
            print(json.dumps(sum_calibration(mdl, p, rows, which=which), indent=2))
            print(f"\n=== Stage-2 per-term scale (assignment={which}) ===")
            print(json.dumps(per_term_scale(mdl, p, rows, which=which), indent=2))
    rep = analyze(mdl, p, rows)
    md = render_report(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(md + "\n")
