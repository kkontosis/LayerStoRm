"""Phase 1.6 (TD-EPM-DRAFT-FEATURES) — FULL feature fusion + interaction
analysis for the recur<=16 target, offline (GPU-2/CPU), no re-collect.

Phases 1 / 1.5 tested feature GROUPS separately (hiddens 0.742, occurrence
0.746, gate+rank+occ 0.767) — all short of the AUC~0.9 the eviction bridge
needs.  This closes the cheap path by fusing EVERYTHING in one predictor and
mining interactions:

  features per committed+labeled routed slot:
    hidden_logit  — the trained direct_h16 head's recur logit (the tractable
                    learned projection of the 5-tap draft hiddens; concatenating
                    raw 30720-dim hiddens per slot is infeasible)
    gate          — labels_top_w (unbiased gate weight)
    rank          — slot index in labels_top_ids (noaux_tc selection rank)
    occ_prev, log1p(trail), inv_recency, log1p(recency), hitbefore  (occurrence)
    layer_frac, kpos_frac  (context)
  engineered crosses (theory: the one-shot class may live in an interaction):
    rank*gate, gate*trail, hidden*rank, hidden*gate

  models: convex logistic (trustworthy) | LightGBM GBDT (auto feature crosses,
  best-in-class heterogeneous tabular) | regularized MLP (hidden+tabular).
  interaction/dim: PCA top-components' AUC + GBDT gain importance.

GATE: any model >=0.85 AUC -> re-run evict_sim (GO); else the cheap path is
CONCLUSIVELY closed — report the number + importance verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import dataset as ds, glm_router, model as M, recurrence as rc
from .train_recurrence import (auc, average_precision, recall_prec_at_budget,
                               fit_logistic, apply_logistic)

RAW = ["hidden_logit", "gate", "rank", "occ_prev", "l1_trail", "inv_rec",
       "l1_rec", "hitbefore", "layer_f", "kpos_f"]


def collect(shard_dir, index, keys, mdl, device, stride, max_shards=None,
            batch_blocks=16, want_coords=False):
    shard_dir = Path(shard_dir)
    keys = set(int(k) for k in keys)
    J = len(index["moe_layers"])
    cols = {c: [] for c in ("hidden_logit", "gate", "rank", "occ_prev",
                            "trail", "recency", "label", "layer", "kpos")}
    if want_coords:
        cols.update({"seq": [], "pos": [], "expert": [], "nextdist": []})
    sis = [si for si, sh in enumerate(index["shards"])
           if keys.intersection(int(x) for x in sh["seq_keys"])]
    if max_shards:
        sis = sis[:max_shards]
    mdl.eval()
    with torch.no_grad():
        for si in sis:
            z = np.load(shard_dir / index["shards"][si]["path"])
            sd = rc.load_recur_sidecar(shard_dir, si)
            sk = z["seq_key"].astype(np.int64); ap = z["anchor_pos"].astype(np.int64)
            sel = np.nonzero(np.isin(sk, list(keys)))[0]
            feat = z["features_bf16"]; ids = z["labels_top_ids"]
            tw = z["labels_top_w"]; recur = sd["recur16"]
            prev = sd["occ_prev"]; trail = sd["occ_trail"]
            recency = sd["occ_recency"]; nxt = sd["nextdist"]
            for b0 in range(0, len(sel), batch_blocks):
                rows = sel[b0:b0 + batch_blocks]
                gmax = int(z["gamma"][rows].max())
                f = glm_router.bf16_bits_to_f32(feat[rows, :gmax])
                logit = mdl(torch.from_numpy(np.ascontiguousarray(f)
                                             ).to(device))          # [b,g,J,E]
                idt = torch.from_numpy(ids[rows, :gmax].astype(np.int64)
                                       ).to(device).clamp_min(0)
                hl = torch.gather(logit, -1, idt).cpu().numpy()      # [b,g,J,K]
                rec = recur[rows, :gmax]
                m = rec >= 0
                if stride > 1:
                    flat = np.cumsum(m.reshape(-1)) - 1
                    m = m & ((flat.reshape(m.shape) % stride) == 0)
                B, G, Jc, K = rec.shape
                kk = np.broadcast_to(np.arange(K), rec.shape)
                jj = np.broadcast_to(np.arange(Jc)[None, None, :, None], rec.shape)
                kg = np.broadcast_to(np.arange(G)[None, :, None, None], rec.shape)
                cols["hidden_logit"].append(hl[m].astype(np.float32))
                cols["gate"].append(tw[rows, :gmax][m].astype(np.float32))
                cols["rank"].append(kk[m].astype(np.int8))
                cols["occ_prev"].append(prev[rows, :gmax][m].astype(np.int8))
                cols["trail"].append(trail[rows, :gmax][m].astype(np.int16))
                cols["recency"].append(recency[rows, :gmax][m].astype(np.int16))
                cols["label"].append(rec[m].astype(np.int8))
                cols["layer"].append(jj[m].astype(np.int16))
                cols["kpos"].append(kg[m].astype(np.int8))
                if want_coords:
                    sq = np.broadcast_to(sk[rows][:, None, None, None], rec.shape)
                    pos = (ap[rows][:, None, None, None]
                           + np.arange(G)[None, :, None, None])
                    cols["seq"].append(sq[m].astype(np.int64))
                    cols["pos"].append(np.broadcast_to(pos, rec.shape)[m
                                                                       ].astype(np.int32))
                    cols["expert"].append(ids[rows, :gmax][m].astype(np.int16))
                    cols["nextdist"].append(nxt[rows, :gmax][m].astype(np.int16))
    return {c: np.concatenate(v) for c, v in cols.items()}


def raw_matrix(d):
    """[N, len(RAW)] float32 in RAW order (GBDT/PCA use this)."""
    rc_ = d["recency"].astype(np.float32)
    cols = {
        "hidden_logit": d["hidden_logit"],
        "gate": d["gate"],
        "rank": d["rank"].astype(np.float32),
        "occ_prev": d["occ_prev"].astype(np.float32),
        "l1_trail": np.log1p(d["trail"].astype(np.float32)),
        "inv_rec": np.where(rc_ >= 0, 1.0 / np.maximum(rc_, 1.0), 0.0),
        "l1_rec": np.where(rc_ >= 0, np.log1p(rc_), 0.0),
        "hitbefore": (rc_ >= 0).astype(np.float32),
        "layer_f": d["layer"].astype(np.float32) / 75.0,
        "kpos_f": d["kpos"].astype(np.float32) / 15.0,
    }
    return np.stack([cols[n] for n in RAW], axis=1).astype(np.float32)


def with_crosses(X):
    """Append engineered crosses + rank one-hot for the linear/MLP models."""
    i = {n: k for k, n in enumerate(RAW)}
    hl, gate, rank = X[:, i["hidden_logit"]], X[:, i["gate"]], X[:, i["rank"]]
    trail = X[:, i["l1_trail"]]
    cr = np.stack([rank * gate, gate * trail, hl * rank, hl * gate], axis=1)
    oh = np.zeros((len(X), 8), np.float32)
    oh[np.arange(len(X)), X[:, i["rank"]].astype(np.int64).clip(0, 7)] = 1
    return np.concatenate([X, cr, oh], axis=1).astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d, h=128, p=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.LayerNorm(h), torch.nn.GELU(),
            torch.nn.Dropout(p), torch.nn.Linear(h, h), torch.nn.GELU(),
            torch.nn.Dropout(p), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xev, device, steps=5000, bs=65536, lr=3e-4):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xn = ((Xtr - mu) / sd).astype(np.float32)
    Xe = ((Xev - mu) / sd).astype(np.float32)
    mdl = MLP(Xn.shape[1]).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xn).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    g = torch.Generator(device=device).manual_seed(0)
    n = len(yt)
    mdl.train()
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g, device=device)
        loss = bce(mdl(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
    mdl.eval()
    with torch.no_grad():
        out = []
        Xte = torch.from_numpy(Xe).to(device)
        for i in range(0, len(Xe), 1 << 20):
            out.append(torch.sigmoid(mdl(Xte[i:i + (1 << 20)])).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hidden-model", required=True)
    ap.add_argument("--routers", required=True)
    ap.add_argument("--out", default="build/recur/fusion")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-shards", type=int, default=40)
    ap.add_argument("--held-shards", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--dump-seqs", type=int, default=24)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(cfg["data"]["shards"]); index = ds.load_index(shard_dir)
    bank = glm_router.RouterBank.from_npz(args.routers)
    mdl, _ = M.load_model(args.hidden_model, bank, args.device)
    seq_keys = sorted({int(x) for sh in index["shards"] for x in sh["seq_keys"]})
    tr_k, he_k = ds.sequence_split(seq_keys, cfg["data"]["held_out_fraction"],
                                   cfg["data"]["split_seed"])

    cache = out / f"fus_ts{args.train_shards}_hs{args.held_shards}_s{args.stride}.npz"
    if cache.is_file():
        z = np.load(cache)
        tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
        ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
        print("loaded cache", cache, flush=True)
    else:
        tr = collect(shard_dir, index, tr_k, mdl, args.device, args.stride,
                     max_shards=args.train_shards)
        ev = collect(shard_dir, index, he_k, mdl, args.device, args.stride,
                     max_shards=args.held_shards)
        np.savez(cache, **{f"tr_{k}": v for k, v in tr.items()},
                 **{f"ev_{k}": v for k, v in ev.items()})
        print("cached", cache, flush=True)

    Xtr, Xev = raw_matrix(tr), raw_matrix(ev)
    ytr = (tr["label"] > 0).astype(np.int64); yev = ev["label"] > 0
    base = float(yev.mean())
    print(f"train {ytr.size:,}  held {yev.size:,}  base {base:.4f}", flush=True)

    results = {}

    # single-feature AUC (what is load-bearing?)
    print("single-feature held-out AUC:", flush=True)
    for k, n in enumerate(RAW):
        a = auc(Xev[:, k], yev)
        results.setdefault("single", {})[n] = a
        print(f"  {n:14s} {a:.4f}", flush=True)

    def report(name, p):
        a = auc(p, yev); apv = average_precision(p, yev)
        r, pr = recall_prec_at_budget(p, yev, base)
        results[name] = {"auc": a, "ap": apv, "recall@budget": r,
                         "precision@budget": pr}
        print(f"  {name:22s} AUC {a:.4f}  AP {apv:.4f}", flush=True)
        return a

    print("FUSED models:", flush=True)
    # 1. convex logistic on fused + crosses
    Xtrc, Xevc = with_crosses(Xtr), with_crosses(Xev)
    li = np.random.default_rng(0).permutation(ytr.size)[:3_000_000]
    pk = fit_logistic(Xtrc[li], ytr[li], steps=800, lr=0.5)
    report("logistic_fused+crosses", apply_logistic(pk, Xevc))

    # 2. LightGBM on raw fused (auto crosses)
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=RAW)
    params = dict(objective="binary", metric="auc", num_leaves=127,
                  learning_rate=0.05, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200,
                  verbose=-1, num_threads=8)
    gbm = lgb.train(params, dtr, num_boost_round=400)
    pgbm = gbm.predict(Xev)
    a_gbm = report("lightgbm_fused", pgbm)
    imp = dict(zip(RAW, gbm.feature_importance(importance_type="gain")))
    results["gbm_importance_gain"] = {k: float(v) for k, v in imp.items()}
    print("  GBDT gain importance:", {k: round(v) for k, v in
          sorted(imp.items(), key=lambda x: -x[1])}, flush=True)

    # 3. regularized MLP (hidden + tabular + crosses)
    p_mlp = train_mlp(Xtrc, ytr, Xevc, args.device)
    a_mlp = report("mlp_fused+crosses", p_mlp)

    # PCA: do the top linear components carry signal beyond occurrence?
    mu = Xtr.mean(0); sdv = Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sdv
    C = np.cov(Xs[li[:500000]].T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)
    Xevs = (Xev - mu) / sdv
    pca_auc = []
    for c in order[:5]:
        proj = Xevs @ V[:, c]
        pca_auc.append(round(float(max(auc(proj, yev), auc(-proj, yev))), 4))
    results["pca_top5_component_auc"] = pca_auc
    results["pca_top5_explained"] = [round(float(w[c] / w.sum()), 3)
                                     for c in order[:5]]
    print("  PCA top-5 component AUC:", pca_auc, flush=True)

    best = max(a_gbm, a_mlp, results["logistic_fused+crosses"]["auc"])
    results["best_auc"] = best; results["base_rate"] = base
    results["gate_0p85"] = best >= 0.85
    json.dump(results, open(out / "fusion_eval.json", "w"), indent=1)
    print(f"\nBEST FUSED AUC {best:.4f}  (0.85 gate: "
          f"{'PASS -> run evict_sim' if best >= 0.85 else 'FAIL -> cheap path closed'})",
          flush=True)

    # dump best model's held predictions for evict_sim IF it cleared the bar
    if best >= 0.85:
        dump_keys = sorted(he_k)[:args.dump_seqs]
        dd = collect(shard_dir, index, dump_keys, mdl, args.device, 1,
                     want_coords=True)
        Xd = raw_matrix(dd)
        prob = (gbm.predict(Xd) if a_gbm == best
                else apply_logistic(pk, with_crosses(Xd)))
        np.savez(out / "preds_fused.npz", seq=dd["seq"], pos=dd["pos"],
                 layer=dd["layer"], expert=dd["expert"],
                 prob=prob.astype(np.float32), label=dd["label"],
                 occ_prev=dd["occ_prev"], nextdist=dd["nextdist"])
        print("dumped preds_fused.npz for evict_sim", flush=True)


if __name__ == "__main__":
    main()
