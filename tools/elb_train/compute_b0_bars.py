"""Compute the EPM-5 b0_prev bars (numpy-vectorized, fast at 100k+ blocks).

The metrics.EvalAccumulator per-cell Python set path is O(blocks x G x J)
set intersections — impractical at 105k blocks. This computes the same
quantities (per-(k,j) recall@8 + per-(block,j) union coverage) with numpy
per shard, over a given sequence split, using the prev-membership
sidecars (dataset.build_prev_membership_sidecars) as the b0_prev
prediction.

  recall@8[k,j]   = |prev_top8 ∩ true_top8| / |true_top8|   (labeled cells)
  union_cov[j]    = |∪_k prev ∩ ∪_k true| / |∪_k true|      (per block, then
                    n-weighted mean over blocks) — the M4 "prev-union
                    coverage" deployed bar (report ALL and deep L58-77).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import dataset  # noqa: E402


def _recall_cells(pred: np.ndarray, true: np.ndarray):
    """pred/true [...,K] int32 (-1 pad) -> (inter_count, n_true) [...]. """
    # match[...,a,b] = pred[...,a] == true[...,b] and true valid
    m = (pred[..., :, None] == true[..., None, :]) & (true[..., None, :] >= 0)
    inter = m.any(axis=-2).sum(axis=-1)          # true ids matched by any pred
    ntrue = (true >= 0).sum(axis=-1)
    return inter, ntrue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    SD = a.shards
    idx = dataset.load_index(SD)
    ml = np.asarray(idx["moe_layers"], np.int32)
    J = len(ml); G = idx["max_gamma"]
    deep = np.array([i for i, l in enumerate(ml)
                     if a.deep_lo <= l <= a.deep_hi])
    allk = sorted({k for sh in idx["shards"] for k in sh["seq_keys"]})
    _, held = dataset.sequence_split(allk, a.held_fraction, seed=a.split_seed)
    heset = set(held)

    # recall accumulators (per position k, per layer j)
    inter_kj = np.zeros((G, J)); ntrue_kj = np.zeros((G, J)); ncell_kj = np.zeros((G, J))
    # union accumulators (per layer j)
    ucov_sum = np.zeros(J); ucov_n = np.zeros(J, np.int64)
    nblk = 0
    t = time.time()
    for si, sh in enumerate(idx["shards"]):
        if not heset & set(sh["seq_keys"]):
            continue
        z = np.load(f"{SD}/{sh['path']}")
        prev = np.load(f"{SD}/prev_top_ids_{si:05d}.npy")
        keep = np.isin(z["seq_key"].astype(np.int64), held)
        rows = np.nonzero(keep)[0]
        P = prev[rows]; T = z["labels_top_ids"][rows]      # [b,G,J,K]
        lm = z["label_mask"][rows]                          # [b,G]
        # recall per (b,g,j)
        inter, ntrue = _recall_cells(P, T)                  # [b,G,J]
        valid = lm[:, :, None] & (ntrue > 0)                # [b,G,J]
        rc = np.where(ntrue > 0, inter / np.maximum(ntrue, 1), 0.0)
        inter_kj += np.where(valid, rc, 0.0).sum(0)
        ncell_kj += valid.sum(0)
        # union coverage per (b,j): union over positions k (labeled only)
        b = P.shape[0]
        for bi in range(b):
            lab = np.nonzero(lm[bi])[0]
            if lab.size == 0:
                continue
            for jj in range(J):
                tu = np.unique(T[bi, lab, jj])
                tu = tu[tu >= 0]
                if tu.size == 0:
                    continue
                pu = np.unique(P[bi, lab, jj])
                pu = pu[pu >= 0]
                cov = np.intersect1d(pu, tu, assume_unique=True).size / tu.size
                ucov_sum[jj] += cov; ucov_n[jj] += 1
        nblk += b

    recall_kj = np.where(ncell_kj > 0, inter_kj / np.maximum(ncell_kj, 1), np.nan)
    # overall recall@8 = cell-count-weighted mean over all (k,j)
    r8 = float(np.nansum(inter_kj) / np.nansum(ncell_kj))
    pp = [float(np.nansum(inter_kj[k]) / max(1, np.nansum(ncell_kj[k])))
          for k in range(G)]
    uc = np.where(ucov_n > 0, ucov_sum / np.maximum(ucov_n, 1), np.nan)
    uc_all = float(np.nansum(ucov_sum) / np.nansum(ucov_n))
    deep_uc = float(np.nansum(ucov_sum[deep]) / np.nansum(ucov_n[deep]))
    out = {
        "held_seqs": len(held), "blocks": nblk,
        "b0prev_recall8": r8,
        "prev_union_all": uc_all,
        "prev_union_deep": deep_uc,
        "deep_layers": [int(ml[i]) for i in deep],
        "per_position_recall8": pp,
        "seconds": round(time.time() - t),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "per_position_recall8"}, indent=1))
    print("per-position recall@8:", " ".join(f"{x:.3f}" for x in pp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
