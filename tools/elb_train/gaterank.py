"""Phase 1.5 (TD-EPM-DRAFT-FEATURES) — gate-weight + top-K-rank recurrence
arms, WITHOUT re-collection.

Phase 1 arms (draft_hidden, occ_*, lru_recency) all plateaued at ~0.74 AUC
for recur<=16.  POLICY_LAB finding #3 named the missing features:
per-expert GATE WEIGHT + top-K RANK.  They are ALREADY in the corpus:
`labels_top_w[G,J,K]` (unbiased gate weight of each routed expert) and the
slot index in `labels_top_ids[G,J,K]` (noaux_tc selection rank, 0=strongest
..7=weakest).  EDA confirms strong monotone recurrence signal in BOTH
(rank: 0.788->0.562; gate-weight decile: 0.610->0.824) — and they are
independent (labels_top_w is NOT slot-sorted).  These are causal (known at
route time) so they are deployable eviction features.

Arms (small MLP, BCE, GPU-2), all scored on the SAME held-out split vs the
Phase-1 bars (occ_trail 0.745 / draft_hidden 0.742 AUC):
  occ        : [occ_prev, log1p(trail), 1/recency, hitbefore]  (Phase-1 ref)
  gaterank   : [gate_weight, rank one-hot(8)]  (the NEW pure test)
  gr_occ     : gaterank + occ  (all cheap causal tabular features)
  gr_occ_ctx : gr_occ + layer/pos context
Dumps held-out predictions (evict_sim schema) for the system-side proxy.

NOTE: the small MLP is optimization-UNSTABLE on these low-dim tabular
features (gate outliers + collinear rank one-hot can diverge it below the
raw-feature AUC); each arm therefore also fits a CONVEX logistic (cannot
diverge) and the better-generalizing of the two is used/reported. The
logistic numbers are the trustworthy reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import dataset as ds, recurrence as rc
from .train_recurrence import auc, average_precision, recall_prec_at_budget

FEATS = {
    "occ": ["occ_prev", "l1_trail", "inv_rec", "hitbefore"],
    "gaterank": ["gate", "rank_oh"],
    "gr_occ": ["gate", "rank_oh", "occ_prev", "l1_trail", "inv_rec",
               "hitbefore"],
    "gr_occ_ctx": ["gate", "rank_oh", "occ_prev", "l1_trail", "inv_rec",
                   "hitbefore", "layer_f", "kpos_f"],
}


def collect(shard_dir, index, keys, stride: int, max_shards=None,
            want_coords=False):
    """Gather per committed+labeled slot: scalar features + label (+ coords
    for the evict_sim dump).  Subsampled by `stride` (deterministic slot
    index modulo).  Fully vectorized per shard."""
    shard_dir = Path(shard_dir)
    keys = set(int(k) for k in keys)
    cols: dict[str, list] = {c: [] for c in
                             ("gate", "rank", "occ_prev", "trail", "recency",
                              "label", "layer", "kpos")}
    if want_coords:
        cols.update({"seq": [], "pos": [], "expert": [], "nextdist": []})
    sis = [si for si, sh in enumerate(index["shards"])
           if keys.intersection(int(x) for x in sh["seq_keys"])]
    if max_shards:
        sis = sis[:max_shards]
    for si in sis:
        z = np.load(shard_dir / index["shards"][si]["path"])
        sd = rc.load_recur_sidecar(shard_dir, si)
        sk = z["seq_key"].astype(np.int64)
        ap = z["anchor_pos"].astype(np.int64)
        sel = np.nonzero(np.isin(sk, list(keys)))[0]
        if sel.size == 0:
            continue
        rec = sd["recur16"][sel]                       # [n,G,J,K]
        n, G, J, K = rec.shape
        m = rec >= 0
        if stride > 1:                                 # deterministic thinning
            flat = np.cumsum(m.reshape(-1)) - 1
            m = m & ((flat.reshape(m.shape) % stride) == 0)
        tw = z["labels_top_w"][sel]; ids = z["labels_top_ids"][sel]
        # coordinate grids
        kk = np.broadcast_to(np.arange(K), rec.shape)
        jj = np.broadcast_to(np.arange(J)[None, None, :, None], rec.shape)
        kg = np.broadcast_to(np.arange(G)[None, :, None, None], rec.shape)
        cols["gate"].append(tw[m].astype(np.float32))
        cols["rank"].append(kk[m].astype(np.int8))
        cols["occ_prev"].append(sd["occ_prev"][sel][m].astype(np.int8))
        cols["trail"].append(sd["occ_trail"][sel][m].astype(np.int16))
        cols["recency"].append(sd["occ_recency"][sel][m].astype(np.int16))
        cols["label"].append(rec[m].astype(np.int8))
        cols["layer"].append(jj[m].astype(np.int16))
        cols["kpos"].append(kg[m].astype(np.int8))
        if want_coords:
            sq = np.broadcast_to(sk[sel][:, None, None, None], rec.shape)
            pos = (ap[sel][:, None, None, None]
                   + np.arange(G)[None, :, None, None])
            pos = np.broadcast_to(pos, rec.shape)
            cols["seq"].append(sq[m].astype(np.int64))
            cols["pos"].append(pos[m].astype(np.int32))
            cols["expert"].append(ids[m].astype(np.int16))
            cols["nextdist"].append(sd["nextdist"][sel][m].astype(np.int16))
    return {c: np.concatenate(v) for c, v in cols.items()}


def make_features(d, names, J=75, G=15):
    parts = []
    for n in names:
        if n == "gate":
            parts.append(d["gate"][:, None])
        elif n == "rank_oh":
            oh = np.zeros((d["rank"].size, 8), np.float32)
            oh[np.arange(d["rank"].size), d["rank"].astype(np.int64)] = 1
            parts.append(oh)
        elif n == "occ_prev":
            parts.append(d["occ_prev"].astype(np.float32)[:, None])
        elif n == "l1_trail":
            parts.append(np.log1p(d["trail"].astype(np.float32))[:, None])
        elif n == "inv_rec":
            rc_ = d["recency"].astype(np.float32)
            parts.append(np.where(rc_ >= 0, 1.0 / np.maximum(rc_, 1.0),
                                  0.0)[:, None])
        elif n == "hitbefore":
            parts.append((d["recency"] >= 0).astype(np.float32)[:, None])
        elif n == "layer_f":
            parts.append((d["layer"].astype(np.float32) / J)[:, None])
        elif n == "kpos_f":
            parts.append((d["kpos"].astype(np.float32) / G)[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_arm(Xtr, ytr, device, steps=4000, bs=65536, lr=3e-4):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xn = (Xtr - mu) / sd
    d = Xn.shape[1]
    mdl = MLP(d).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xn).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(yt)
    g = torch.Generator(device=device).manual_seed(0)
    mdl.train()
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g, device=device)
        loss = bce(mdl(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()
    return mdl, mu, sd


@torch.no_grad()
def predict_arm(mdl, mu, sd, X, device):
    Xn = (X - mu) / sd
    out = []
    for i in range(0, len(Xn), 1 << 20):
        xb = torch.from_numpy(Xn[i:i + (1 << 20)]).to(device)
        out.append(torch.sigmoid(mdl(xb)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="build/recur/gaterank")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--eval-stride", type=int, default=4)
    ap.add_argument("--dump-seqs", type=int, default=24,
                    help="held seqs to dump full predictions for evict_sim")
    ap.add_argument("--arms", default="occ,gaterank,gr_occ,gr_occ_ctx")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(cfg["data"]["shards"])
    index = ds.load_index(shard_dir)
    seq_keys = sorted({int(x) for sh in index["shards"]
                       for x in sh["seq_keys"]})
    train_keys, held_keys = ds.sequence_split(
        seq_keys, cfg["data"]["held_out_fraction"], cfg["data"]["split_seed"])
    print(f"split {len(train_keys)}/{len(held_keys)} seqs", flush=True)

    cache = out / f"tab_ts{args.train_stride}_es{args.eval_stride}.npz"
    if cache.is_file():
        z = np.load(cache)
        tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
        ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
        print("loaded tabular cache", cache, flush=True)
    else:
        tr = collect(shard_dir, index, train_keys, args.train_stride)
        ev = collect(shard_dir, index, held_keys, args.eval_stride)
        np.savez(cache, **{f"tr_{k}": v for k, v in tr.items()},
                 **{f"ev_{k}": v for k, v in ev.items()})
        print("cached tabular", cache, flush=True)
    y_tr = (tr["label"] > 0).astype(np.int64)
    y_ev = ev["label"] > 0
    base = float(y_ev.mean())
    print(f"train {y_tr.size:,} slots, held {y_ev.size:,} slots, base "
          f"{base:.4f}", flush=True)

    table = {}
    # raw-feature references (no training): the Phase-1 bars, recomputed here
    for name, sc in (("occ_trail", ev["trail"].astype(np.float32)),
                     ("gate_raw", ev["gate"]),
                     ("rank_raw", -ev["rank"].astype(np.float32))):
        table[name] = {"auc": auc(sc, y_ev), "ap": average_precision(sc, y_ev)}
        print(f"  [ref] {name:12s} AUC {table[name]['auc']:.4f} "
              f"AP {table[name]['ap']:.4f}", flush=True)

    from .train_recurrence import fit_logistic, apply_logistic
    # convex-logistic subsample (instability-proof lower bound)
    li = np.random.default_rng(0).permutation(y_tr.size)[:2_000_000]

    arms = args.arms.split(",")
    trained = {}
    for arm in arms:
        Xtr = make_features(tr, FEATS[arm])
        Xev = make_features(ev, FEATS[arm])
        pk = fit_logistic(Xtr[li], y_tr[li], steps=600, lr=0.5)
        plog = apply_logistic(pk, Xev)
        alog = auc(plog, y_ev); aplog = average_precision(plog, y_ev)
        mdl, mu, sd = train_arm(Xtr, y_tr, args.device)
        p = predict_arm(mdl, mu, sd, Xev, args.device)
        a = auc(p, y_ev); apv = average_precision(p, y_ev)
        # keep whichever generalizes better for the system-side dump
        use_mlp = a >= alog
        pbest = p if use_mlp else plog
        r, pr = recall_prec_at_budget(pbest, y_ev, base)
        table[arm] = {"auc_mlp": a, "ap_mlp": apv, "auc_log": alog,
                      "ap_log": aplog, "recall@budget": r,
                      "precision@budget": pr, "best": "mlp" if use_mlp
                      else "logistic"}
        trained[arm] = (mdl, mu, sd, pk, use_mlp)
        print(f"  {arm:12s} MLP AUC {a:.4f}/AP {apv:.4f}  LOG AUC "
              f"{alog:.4f}/AP {aplog:.4f}  best={table[arm]['best']}",
              flush=True)

    json.dump({"base_rate": base, "held_slots": int(y_ev.size),
               "table": table}, open(out / "gaterank_eval.json", "w"),
              indent=1)

    # dump full held predictions for evict_sim (best arm = last in list)
    best = arms[-1]
    dump_keys = sorted(held_keys)[:args.dump_seqs]
    dd = collect(shard_dir, index, dump_keys, stride=1, want_coords=True)
    Xd = make_features(dd, FEATS[best])
    mdl, mu, sd, pk, use_mlp = trained[best]
    prob = (predict_arm(mdl, mu, sd, Xd, args.device) if use_mlp
            else apply_logistic(pk, Xd).astype(np.float32))
    np.savez(out / f"preds_{best}.npz", seq=dd["seq"], pos=dd["pos"],
             layer=dd["layer"], expert=dd["expert"], prob=prob.astype(np.float32),
             label=dd["label"], occ_prev=dd["occ_prev"],
             nextdist=dd["nextdist"])
    print(f"dumped {prob.size:,} slots ({best}) -> {out}/preds_{best}.npz",
          flush=True)


if __name__ == "__main__":
    main()
