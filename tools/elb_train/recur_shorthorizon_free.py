#!/usr/bin/env python3
"""TASK 2 — FREE short-horizon recurrence predictors on the EPM-5 corpus.

Target: recur@k[P,j,e] = 1 iff routed expert e (committed pos P, MoE layer j)
recurs at layer j within the next k committed tokens (the PROTECT-on-GPU set;
0 = safe-to-offload). Derived from the existing recur_%05d.npz sidecars
(nextdist), so NO retraining/recollection — k just re-thresholds nextdist.

FREE predictors scored at k in {2,4,16}:
  b0_prev  = occ_prev  (1 iff e routed at layer j at the PREV committed pos) —
             the zero-model temporal-locality signal the ledger leans on.
  recency  = -occ_recency (recent route => likely recur; LRU-implicit) — free.
  trail    = occ_trail   (trailing-16 route count; occurrence freq) — free.
All three are computable with zero model. Reports ROC-AUC + the offload-decision
operating metrics (protect-class + safe/offload-class precision/recall).
"""
import argparse
import glob
import os
import numpy as np

SD = "/srv/models/epm5-corpus/shards"


def auc_from_scores(score, label, nbins=512):
    """Fast histogram ROC-AUC. label in {0,1}, score float (higher=>positive)."""
    lo, hi = score.min(), score.max()
    if hi <= lo:
        return 0.5
    b = np.clip(((score - lo) / (hi - lo) * (nbins - 1)).astype(np.int64),
                0, nbins - 1)
    pos = np.bincount(b[label == 1], minlength=nbins).astype(np.float64)
    neg = np.bincount(b[label == 0], minlength=nbins).astype(np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    # AUC = sum over bins of (neg-below + 0.5*neg-in-bin)/N weighted by pos frac
    cneg = np.cumsum(neg) - neg            # neg strictly below each bin
    auc = (pos * (cneg + 0.5 * neg)).sum() / (P * N)
    return auc


def _logistic_split_auc(X, y, iters=300, lr=0.5, sub=4_000_000):
    """Standardized logistic regression, 50/50 seq-agnostic split (b0_prev/occ
    features are zero-model so block-split leakage is negligible), GD in numpy.
    Subsamples to `sub` rows for the fit; evals AUC on the held half."""
    rng = np.random.default_rng(0)
    n = X.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    tr, te = perm[:half], perm[half:]
    Xtr, ytr = X[tr], y[tr].astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    if Xtr.shape[0] > sub:
        s = rng.choice(Xtr.shape[0], sub, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    w = np.zeros(X.shape[1], np.float32)
    b = 0.0
    for _ in range(iters):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - ytr
        w -= lr * (Xtr.T @ g / Xtr.shape[0])
        b -= lr * g.mean()
    Xte = (X[te] - mu) / sd
    score = Xte @ w + b
    return auc_from_scores(score, y[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=48)
    ap.add_argument("--ks", default="2,4,16")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    files = sorted(glob.glob(os.path.join(SD, "recur_*.npz")))[:args.nshards]

    nd_all, prev_all, rec_all, trail_all = [], [], [], []
    for f in files:
        z = np.load(f)
        r16 = z["recur16"]
        valid = (r16 >= 0).ravel()
        nd_all.append(z["nextdist"].ravel()[valid])
        prev_all.append(z["occ_prev"].ravel()[valid].astype(np.int8))
        rec_all.append(z["occ_recency"].ravel()[valid].astype(np.int32))
        trail_all.append(z["occ_trail"].ravel()[valid].astype(np.int16))
    nd = np.concatenate(nd_all)
    prev = np.concatenate(prev_all)
    recency = np.concatenate(rec_all)
    trail = np.concatenate(trail_all)
    n = nd.size
    print(f"shards={len(files)} labeled routed slots={n:,}")
    # free scores (higher => predict RECUR/protect)
    prev_s = prev.astype(np.float32)
    # recency: -1 = first-ever (never routed before => predict NOT recur/lowest);
    # else distance since last route (small=recent=>recur). score = 1/(1+dist),
    # first-ever gets 0.
    rec_s = np.where(recency < 0, 0.0, 1.0 / (1.0 + recency)).astype(np.float32)
    trail_s = trail.astype(np.float32)

    print("\ntarget recur@k = protect-on-GPU (1); complement = safe-to-offload (0)")
    print("k   base_recur%  safe%   | b0_prev: AUC  protРec protPrec  safeRec safePrec")
    print("                          |   recency_AUC  trail_AUC")
    for k in ks:
        y = ((nd >= 1) & (nd <= k)).astype(np.int8)   # 1=recurs within k
        base = y.mean()
        # b0_prev as a binary decision: predict protect iff occ_prev==1
        tp = int(((prev == 1) & (y == 1)).sum())
        fp = int(((prev == 1) & (y == 0)).sum())
        fn = int(((prev == 0) & (y == 1)).sum())
        tn = int(((prev == 0) & (y == 0)).sum())
        prot_rec = tp / max(tp + fn, 1)          # recall of the recur/protect set
        prot_prec = tp / max(tp + fp, 1)
        # offload decision = predict SAFE iff occ_prev==0; safe class is y==0
        safe_rec = tn / max(tn + fp, 1)          # of truly-safe, how many we offload
        safe_prec = tn / max(tn + fn, 1)         # of offloaded, how many truly safe
        auc_prev = auc_from_scores(prev_s, y)
        auc_rec = auc_from_scores(rec_s, y)
        auc_trail = auc_from_scores(trail_s, y)
        # combined FREE logistic on [prev, log1p(trail), 1/(1+recency),
        # first_ever] — the ceiling of the free occurrence features (the direct
        # analog of the campaign's occ_logistic; trained/eval on a 50/50 split).
        X = np.stack([prev.astype(np.float32),
                      np.log1p(trail.astype(np.float32)),
                      rec_s,
                      (recency < 0).astype(np.float32)], axis=1)
        auc_comb = _logistic_split_auc(X, y)
        print(f"{k:2d}   {base*100:6.1f}     {(1-base)*100:5.1f}  | "
              f"{auc_prev:.3f}  {prot_rec:.3f}   {prot_prec:.3f}    "
              f"{safe_rec:.3f}   {safe_prec:.3f}")
        print(f"                          |   recency {auc_rec:.3f}   "
              f"trail {auc_trail:.3f}   free-combined-logistic {auc_comb:.3f}")


if __name__ == "__main__":
    main()
