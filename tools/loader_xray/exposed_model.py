#!/usr/bin/env python3
"""EXPOSED-WALL loader cost model (M2) — fit + cross-trace validation.

The current model predicts the RAW transfer window; the engine hides part of it
under resident-expert compute (fetch-overlap, INV-MOE-OVERLAP) and the token
wall depends on WHICH device stalls (critical-path position). M2:

  sub_j      = Σ subxfer of misses executed on device j       (row structure)
  hits_j     = # cached experts executed on device j          (overlap budget)
  exposed_j  = max(0, s_j·sub_j − (o0_j + oc_j·hits_j))       (overlap credit)
  wall       = max_j exposed_j                                 → transfer_wall_us
  total      = max_j (cpw_j·exposed_j) + b0                    → total_us

Params (per M=4): s_j (device scale), o0_j, oc_j (overlap credit), cpw_j
(critical-path weight), b0. Fit: numpy Adam on joint RMSE; subgradient through
the max. Train on one trajectory, validate on the other (different config AND
different routing trace — the honest generalization test).

M2v2 (--form v2, the B>1-forward form — DETHRONE_AFFINITY_IDEAS.md reqs):
  compute_j  = g_c·(a_j·cnt_j + b_j·ceil(cnt_j/P_j))   (batch-step curve KEPT,
               never absorbed into the credit — wrong at B>1 otherwise)
  sat(h)     = hsat_j·(1 − exp(−h/hsat_j))             (SATURATING credit,
               ≈h at small h, bounded at batch-scale hit counts)
  exposed_j  = max(0, s_j·sub_j − o0_j − oc_j·sat(hits_j))
  wall       = max_j exposed_j
  total      = max_j cpw_j·(compute_j + exposed_j) + b0
Valid at arbitrary miss counts (prefill shapes); monotone in sub_/cnt_ given
FIXED hits, so the C++ port stays pinned-tier-only (B&B bound validity).

Usage:
  python3 exposed_model.py --form v2 \
                           --calib <gpu_loader_calibration.json> \
                           --train <model.jsonl>:<trace.csv>:<act|aff> \
                           --val   <model.jsonl>:<trace.csv>:<act|aff>
"""
import argparse
import json

import numpy as np

from joiner import join


def executed_assignment(e, mode):
    oj, j = e.get("oj", -1), e.get("j", -1)
    if mode == "aff" or j < 0:
        return oj
    return oj if (0 <= oj < 4 and e["cached"][oj]) else j


def build(rows, mode, M=4):
    """Per joined row -> (sub[M], hits[M], nmiss[M], y_wall, y_total)."""
    S, H, NM, YW, YT = [], [], [], [], []
    for r in rows:
        real = r.get("real", r)
        if real.get("transfer_wall_us") is None or real.get("total_us") is None:
            continue
        sub = [0.0] * M
        hits = [0] * M
        nm = [0] * M
        for e in r["experts"]:
            g = executed_assignment(e, mode)
            if not 0 <= g < M:
                continue
            if e["cached"][g]:
                hits[g] += 1
            else:
                nm[g] += 1
                sub[g] += e["subxfer_us"][g]
        S.append(sub); H.append(hits); NM.append(nm)
        YW.append(real["transfer_wall_us"]); YT.append(real["total_us"])
    return (np.array(S), np.array(H, dtype=float), np.array(NM, dtype=float),
            np.array(YW), np.array(YT))


def compute_curve(calib, CNT):
    """g_c-free batch-step compute per device: a·c + b·ceil(c/P), c=CNT[n,M]."""
    M = CNT.shape[1]
    out = np.zeros_like(CNT)
    for j in range(M):
        d = calib["devices"][j]["compute"]
        c = CNT[:, j]
        out[:, j] = d["a_us"] * c + d["b_us"] * np.ceil(c / d["P"]) * (c > 0)
    return out


def predict_v2(theta, S, H, COMP):
    M = S.shape[1]
    s    = theta[0:M]
    o0   = theta[M:2*M]
    oc   = theta[2*M:3*M]
    hsat = theta[3*M:4*M]
    cpw  = theta[4*M:5*M]
    g_c  = theta[5*M]
    b0   = theta[5*M + 1]
    sat = hsat * (1.0 - np.exp(-H / np.maximum(hsat, 1e-6)))
    exposed = np.maximum(0.0, S * s - (o0 + oc * sat))
    wall = exposed.max(axis=1)
    total = (cpw * (g_c * COMP + exposed)).max(axis=1) + b0
    return wall, total


def fit_v2(S, H, COMP, YW, YT, iters=5000, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    M = S.shape[1]
    theta = np.concatenate([np.full(M, 0.7), np.zeros(M), np.full(M, 50.0),
                            np.full(M, 8.0), np.full(M, 1.0), [1.0],
                            [YT.mean() - YW.mean()]])
    mw, vw = np.zeros_like(theta), np.zeros_like(theta)
    scale_w = 1.0 / max(YW.std(), 1.0)
    scale_t = 1.0 / max(YT.std(), 1.0)
    for it in range(iters):
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        c = 0.01 * (np.abs(theta) + 1.0)
        def loss(th):
            w, t = predict_v2(th, S, H, COMP)
            return (np.sqrt(np.mean(((w - YW) * scale_w) ** 2))
                    + np.sqrt(np.mean(((t - YT) * scale_t) ** 2)))
        lp, lm = loss(theta + c * delta), loss(theta - c * delta)
        g = (lp - lm) / (2 * c) * delta
        mw = 0.9 * mw + 0.1 * g
        vw = 0.999 * vw + 0.001 * g * g
        theta -= lr * mw / (np.sqrt(vw) + 1e-8) * (np.abs(theta) + 1.0)
        theta[0:M] = np.clip(theta[0:M], 0.05, 3.0)
        theta[M:3*M] = np.maximum(theta[M:3*M], 0.0)
        theta[3*M:4*M] = np.clip(theta[3*M:4*M], 1.0, 64.0)
        theta[4*M:5*M] = np.clip(theta[4*M:5*M], 0.1, 5.0)
        theta[5*M] = np.clip(theta[5*M], 0.0, 3.0)
    return theta


def predict(theta, S, H):
    M = S.shape[1]
    s   = theta[0:M]
    o0  = theta[M:2*M]
    oc  = theta[2*M:3*M]
    cpw = theta[3*M:4*M]
    b0  = theta[4*M]
    exposed = np.maximum(0.0, S * s - (o0 + H * oc))          # [n, M]
    wall = exposed.max(axis=1)
    total = (exposed * cpw).max(axis=1) + b0
    return wall, total


def r2(pred, y):
    ss = np.sum((y - pred) ** 2)
    st = np.sum((y - y.mean()) ** 2) or 1.0
    return 1.0 - ss / st


def fit(S, H, YW, YT, iters=4000, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    M = S.shape[1]
    theta = np.concatenate([np.full(M, 0.7), np.zeros(M), np.full(M, 50.0),
                            np.full(M, 1.0), [YT.mean() - YW.mean()]])
    mw, vw = np.zeros_like(theta), np.zeros_like(theta)
    scale_w = 1.0 / max(YW.std(), 1.0)
    scale_t = 1.0 / max(YT.std(), 1.0)
    eps = 1e-8
    for it in range(iters):
        # numeric gradient via SPSA (cheap, robust through the max/relu kinks)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        c = 0.01 * (np.abs(theta) + 1.0)
        def loss(th):
            w, t = predict(th, S, H)
            return (np.sqrt(np.mean(((w - YW) * scale_w) ** 2))
                    + np.sqrt(np.mean(((t - YT) * scale_t) ** 2)))
        lp, lm = loss(theta + c * delta), loss(theta - c * delta)
        g = (lp - lm) / (2 * c) * delta
        mw = 0.9 * mw + 0.1 * g
        vw = 0.999 * vw + 0.001 * g * g
        theta -= lr * mw / (np.sqrt(vw) + eps) * (np.abs(theta) + 1.0)
        # clamp: scales/credits/weights non-negative
        theta[0:M] = np.clip(theta[0:M], 0.05, 3.0)
        theta[M:3*M] = np.maximum(theta[M:3*M], 0.0)
        theta[3*M:4*M] = np.clip(theta[3*M:4*M], 0.1, 5.0)
    return theta


def baseline_raw(S):
    """The CURRENT model's transfer view: raw serial per-device sum, no overlap."""
    return S.max(axis=1)


def report(tag, form, theta, S, H, COMP, YW, YT):
    if form == "v2":
        w, t = predict_v2(theta, S, H, COMP)
    else:
        w, t = predict(theta, S, H)
    bw = baseline_raw(S)
    print(f"  {tag}: n={len(YW)}")
    print(f"    transfer_wall: {form} R2={r2(w, YW):.3f}   (raw-max R2={r2(bw, YW):.3f})")
    print(f"    total_wall:    {form} R2={r2(t, YT):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", action="append", default=[])
    ap.add_argument("--out", default=None, help="write fitted params JSON")
    ap.add_argument("--form", default="v1", choices=["v1", "v2"])
    ap.add_argument("--calib", default=None,
                    help="calibration JSON (required for --form v2 compute term)")
    args = ap.parse_args()
    calib = json.load(open(args.calib)) if args.calib else None
    if args.form == "v2" and calib is None:
        ap.error("--form v2 requires --calib")

    def load(spec):
        mj, tc, mode = spec.split(":")
        rows, stats = join(mj, tc)
        S, H, NM, YW, YT = build(rows, mode)
        COMP = compute_curve(calib, H + NM) if calib is not None else np.zeros_like(S)
        return S, H, COMP, YW, YT

    S, H, COMP, YW, YT = load(args.train)
    M = S.shape[1]
    if args.form == "v2":
        theta = fit_v2(S, H, COMP, YW, YT)
        print("fitted params (v2):")
        print(f"  s    = {np.round(theta[0:M], 3).tolist()}")
        print(f"  o0   = {np.round(theta[M:2*M], 1).tolist()} us")
        print(f"  oc   = {np.round(theta[2*M:3*M], 1).tolist()} us/hit")
        print(f"  hsat = {np.round(theta[3*M:4*M], 2).tolist()} hits")
        print(f"  cpw  = {np.round(theta[4*M:5*M], 3).tolist()}")
        print(f"  g_c  = {theta[5*M]:.3f}   b0 = {theta[5*M+1]:.1f} us")
    else:
        theta = fit(S, H, YW, YT)
        print("fitted params:")
        print(f"  s   = {np.round(theta[0:M], 3).tolist()}")
        print(f"  o0  = {np.round(theta[M:2*M], 1).tolist()} us")
        print(f"  oc  = {np.round(theta[2*M:3*M], 1).tolist()} us/hit")
        print(f"  cpw = {np.round(theta[3*M:4*M], 3).tolist()}")
        print(f"  b0  = {theta[4*M]:.1f} us")
    report("TRAIN", args.form, theta, S, H, COMP, YW, YT)
    for v in args.val:
        report("VAL  " + v.split(":")[0].split("/")[-1], args.form, theta, *load(v))
    if args.out:
        if args.form == "v2":
            out = {"form": "v2", "s": theta[0:M].tolist(),
                   "o0": theta[M:2*M].tolist(), "oc": theta[2*M:3*M].tolist(),
                   "hsat": theta[3*M:4*M].tolist(),
                   "cpw": theta[4*M:5*M].tolist(),
                   "g_c": float(theta[5*M]), "b0": float(theta[5*M+1])}
        else:
            out = {"form": "v1", "s": theta[0:M].tolist(),
                   "o0": theta[M:2*M].tolist(), "oc": theta[2*M:3*M].tolist(),
                   "cpw": theta[3*M:4*M].tolist(), "b0": float(theta[4*M])}
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
