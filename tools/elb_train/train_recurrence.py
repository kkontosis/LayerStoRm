"""Recurrence predictor training + baseline shoot-out (TD-EPM-DRAFT-FEATURES
Phase 1, steps 2-3 model side).

THE scientific test: do the frozen 5-tap DSpark draft hiddens predict
"expert e recurs at layer j within 16 committed decode tokens" (the
DSP52_EVICT_BRIDGE signal), where OCCURRENCE statistics cannot (POLICY_LAB
finding #3)?

Population: every committed+labeled ROUTED slot on the held-out split
(recur_%05d.npz sidecar recur16 >= 0).  Target: recur16 in {0,1}.  All
methods score the SAME slots; compared by AUC / average-precision / and
recall+precision at a protection budget equal to the base positive rate.

Methods
  base_rate     constant (reference; recall 1.0 @ full protect).
  occ_prev      1 iff e routed at layer j at the previous committed pos
                (b0_prev temporal locality — the free signal the whole
                ledger leans on).
  occ_trail     trailing-16 route count of e at j (occurrence frequency).
  lru_recency   -distance-since-last-route (LRU-implicit: recent => keep).
  occ_logistic  logistic reg on [prev, log1p(trail), 1/recency, hit-before]
                trained on the TRAIN split (finding #3's loser vs LRU).
  draft_hidden  the trained EPM head over the 5-tap hiddens (this tool).

Reuses model.EpmPredictor (direct/router arm, rmsnorm, 5-tap fusion,
W-less BCE) + dataset shard iteration + glm_router RouterBank replicas
(from an existing build/epm5-grid/*/routers.npz — never touches NVFP4).
INV-EPM-SIDE holds by construction (standalone torch tool).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import dataset as ds, glm_router, model as M, recurrence as rc


# ── data ─────────────────────────────────────────────────────────────────────

def _shards_for(index: dict, keys: set[int]) -> list[int]:
    return [si for si, sh in enumerate(index["shards"])
            if keys.intersection(int(x) for x in sh["seq_keys"])]


def iter_batches(shard_dir: Path, index: dict, keys: set[int],
                 batch_blocks: int, rng, shuffle: bool,
                 max_shards: int | None = None):
    """Yield (feat f32[B,G,L,H], ids i64[B,G,J,K], recur i8, prev i8,
    trail i8, recency i16) block batches on the given split.  Loads one
    shard at a time (features are big); shuffles shard + block order.
    max_shards caps the number of shards visited (eval/logistic subsample)."""
    sis = _shards_for(index, keys)
    if shuffle:
        rng.shuffle(sis)
    if max_shards is not None:
        sis = sis[:max_shards]
    for si in sis:
        z = np.load(shard_dir / index["shards"][si]["path"])
        sd = rc.load_recur_sidecar(shard_dir, si)
        if sd is None:
            raise FileNotFoundError(f"recur sidecar missing for shard {si}")
        sk = z["seq_key"].astype(np.int64)
        sel = np.nonzero(np.isin(sk, list(keys)))[0]
        if shuffle:
            rng.shuffle(sel)
        feat = z["features_bf16"]; ids = z["labels_top_ids"]
        recur = sd["recur16"]; prev = sd["occ_prev"]
        trail = sd["occ_trail"]; recency = sd["occ_recency"]
        for b0 in range(0, len(sel), batch_blocks):
            rows = sel[b0:b0 + batch_blocks]
            f = glm_router.bf16_bits_to_f32(feat[rows])          # [B,G,L,H]
            yield (torch.from_numpy(np.ascontiguousarray(f)),
                   ids[rows].astype(np.int64), recur[rows],
                   prev[rows], trail[rows], recency[rows])


# ── metrics (numpy, sklearn-free) ────────────────────────────────────────────

def _avg_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(score: np.ndarray, label: np.ndarray) -> float:
    p = label > 0
    npos = int(p.sum()); nneg = int((~p).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _avg_rank(score.astype(np.float64))
    return float((r[p].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    y = (label[order] > 0).astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    npos = tp[-1]
    if npos == 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / npos
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def recall_prec_at_budget(score: np.ndarray, label: np.ndarray,
                          budget: float) -> tuple[float, float]:
    """Protect the top-`budget` fraction by score; return (recall,
    precision) over the protected set vs the positives (recur==1)."""
    n = len(score)
    k = max(1, int(round(budget * n)))
    thr_idx = np.argpartition(-score, k - 1)[:k]
    prot = np.zeros(n, bool); prot[thr_idx] = True
    p = label > 0
    tp = int((prot & p).sum())
    npos = int(p.sum())
    recall = tp / npos if npos else float("nan")
    prec = tp / int(prot.sum()) if prot.sum() else float("nan")
    return recall, prec


# ── training ─────────────────────────────────────────────────────────────────

def build_recur_model(cfg: dict, index: dict, bank: glm_router.RouterBank,
                      device: str) -> M.EpmPredictor:
    dims = M.Dims.from_index(index)
    # W is unused (recurrence BCE is unweighted); pass a uniform matrix.
    from . import wmap as wmap_mod
    w = wmap_mod.build_wmap({"kind": "uniform"}, dims.max_gamma,
                            dims.moe_layers)
    mdl = M.EpmPredictor(cfg["model"], dims, w, bank, seed=int(cfg["seed"]))
    for buf in mdl.buffers():
        buf.requires_grad_(False)
    return mdl.to(device)


def train(cfg: dict, out: Path, device: str, max_steps: int, log,
          bank: glm_router.RouterBank):
    shard_dir = Path(cfg["data"]["shards"])
    index = ds.load_index(shard_dir)
    seq_keys = sorted({int(x) for sh in index["shards"]
                       for x in sh["seq_keys"]})
    train_keys, held_keys = ds.sequence_split(
        seq_keys, cfg["data"]["held_out_fraction"],
        cfg["data"]["split_seed"])
    log(f"split: {len(train_keys)} train / {len(held_keys)} held seqs "
        f"(seed {cfg['data']['split_seed']})")
    mdl = build_recur_model(cfg, index, bank, device)
    log(f"model: arm={mdl.arm} trainable_params="
        f"{mdl.num_trainable_params():,}")
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]))
    warmup = cfg["optim"]["warmup_steps"]

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))

    rng = np.random.default_rng(int(cfg["seed"]))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    step = 0
    t0 = time.time()
    running = 0.0
    mdl.train()
    while step < max_steps:
        for feat, ids, recur, _pv, _tr, _rc in iter_batches(
                shard_dir, index, set(train_keys),
                cfg["data"]["batch_blocks"], rng, shuffle=True):
            if step >= max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = cfg["optim"]["lr"] * lr_at(step)
            feat = feat.to(device)
            logits = mdl(feat)                       # [B,G,J,E]
            idt = torch.from_numpy(ids).to(device).clamp_min(0)
            slot_logit = torch.gather(logits, -1, idt)  # [B,G,J,K]
            tgt = torch.from_numpy(recur.astype(np.float32)).to(device)
            mask = torch.from_numpy((recur >= 0)).to(device).float()
            loss_el = bce(slot_logit, tgt.clamp(0, 1)) * mask
            denom = mask.sum().clamp_min(1.0)
            loss = loss_el.sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                mdl.parameters(), cfg["optim"]["clip_grad_norm"])
            opt.step()
            running += float(loss.detach())
            step += 1
            if step % cfg.get("log_every", 100) == 0:
                log(f"step {step:5d}/{max_steps} loss "
                    f"{running / cfg.get('log_every', 100):.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} "
                    f"{(time.time() - t0) / step * 1000:.0f} ms/step")
                running = 0.0
    out.mkdir(parents=True, exist_ok=True)
    M.save_model(mdl, str(out / "recur_model"),
                 sidecar_extra={"task": "recurrence", "horizon": rc.RECUR_HORIZON})
    return mdl, index, train_keys, held_keys


# ── evaluation / baseline shoot-out ──────────────────────────────────────────

def collect_eval(mdl, cfg, index, keys, device, log, sub_stride: int = 1,
                 max_shards: int | None = None):
    """Run the model + gather baseline features over `keys`.  Returns a
    dict of per-slot arrays (subsampled by sub_stride): score_<method>,
    label, kpos, layer."""
    shard_dir = Path(cfg["data"]["shards"])
    rng = np.random.default_rng(0)
    J = len(index["moe_layers"])
    cols = {m: [] for m in ("draft_hidden", "occ_prev", "occ_trail",
                            "lru_recency")}
    lab, kpos, lyr, logit_dh = [], [], [], []
    feats_occ = []  # for the logistic (prev, log1p trail, 1/recency, hitbefore)
    mdl.eval()
    with torch.no_grad():
        for feat, ids, recur, prev, trail, recency in iter_batches(
                shard_dir, index, set(keys),
                cfg["data"]["batch_blocks"], rng, shuffle=False,
                max_shards=max_shards):
            logits = mdl(feat.to(device))            # [B,G,J,E]
            idt = torch.from_numpy(ids).to(device).clamp_min(0)
            sl = torch.gather(logits, -1, idt)       # [B,G,J,K] logit
            prob = torch.sigmoid(sl).cpu().numpy()
            B, G, Jc, K = recur.shape
            m = (recur >= 0)
            # positions grid
            kg = np.broadcast_to(np.arange(G)[None, :, None, None],
                                 recur.shape)
            jg = np.broadcast_to(np.arange(Jc)[None, None, :, None],
                                 recur.shape)
            idx = np.nonzero(m)
            # subsample
            if sub_stride > 1:
                keep = (np.arange(idx[0].size) % sub_stride) == 0
                idx = tuple(a[keep] for a in idx)
            lab.append(recur[idx].astype(np.int8))
            kpos.append(kg[idx].astype(np.int8))
            lyr.append(jg[idx].astype(np.int16))
            logit_dh.append(sl.cpu().numpy()[idx].astype(np.float32))
            cols["draft_hidden"].append(prob[idx].astype(np.float32))
            pv = prev[idx].astype(np.float32)
            tr = trail[idx].astype(np.float32)
            rcn = recency[idx].astype(np.float32)
            cols["occ_prev"].append(pv)
            cols["occ_trail"].append(tr)
            # LRU-implicit: more-recent => higher recur prob; never(-1) lowest
            lru = np.where(rcn >= 0, -rcn, -1e9)
            cols["lru_recency"].append(lru.astype(np.float32))
            hitbefore = (rcn >= 0).astype(np.float32)
            inv_rec = np.where(rcn >= 0, 1.0 / np.maximum(rcn, 1.0), 0.0)
            feats_occ.append(np.stack(
                [pv, np.log1p(tr), inv_rec, hitbefore], axis=1))
    out = {m: np.concatenate(v) for m, v in cols.items()}
    out["label"] = np.concatenate(lab)
    out["kpos"] = np.concatenate(kpos)
    out["layer"] = np.concatenate(lyr)
    out["logit_dh"] = np.concatenate(logit_dh)
    out["occ_feats"] = np.concatenate(feats_occ)
    return out


def dump_predictions(mdl, cfg, index, keys, device, out_path: Path, log):
    """Full (no-subsample) held-out predictions WITH coordinates, for the
    offline eviction cache-sim (evict_sim.py).  Writes an npz of parallel
    arrays over every committed+labeled routed slot, in trajectory order
    per sequence: seq, pos (abs committed decode position), layer (0..J-1),
    expert id, model prob, recur16 label, occ_prev, nextpos-distance.
    """
    shard_dir = Path(cfg["data"]["shards"])
    idxmap = {int(x): None for x in keys}
    seqs = rc._seq_timelines(shard_dir, index)  # seq -> committed cells
    # Per (si,row,k) -> (seq, abs_pos, timeline_t)
    coord = {}
    for s, cells in seqs.items():
        if s not in idxmap:
            continue
        for t, (absp, si, row, k, _tids) in enumerate(cells):
            coord[(si, row, k)] = (s, absp, t)
    J = len(index["moe_layers"])
    rng = np.random.default_rng(0)
    cols = {n: [] for n in ("seq", "pos", "layer", "expert", "prob",
                            "label", "occ_prev", "nextdist")}
    mdl.eval()
    with torch.no_grad():
        sis = _shards_for(index, set(keys))
        for si in sis:
            z = np.load(shard_dir / index["shards"][si]["path"])
            sd = rc.load_recur_sidecar(shard_dir, si)
            sk = z["seq_key"].astype(np.int64)
            sel = np.nonzero(np.isin(sk, list(keys)))[0]
            feat = z["features_bf16"]; ids = z["labels_top_ids"]
            recur = sd["recur16"]; prev = sd["occ_prev"]; nxt = sd["nextdist"]
            for row in sel:
                gm = int(z["gamma"][row])
                f = glm_router.bf16_bits_to_f32(feat[row:row + 1, :gm])
                logit = mdl(torch.from_numpy(np.ascontiguousarray(f)
                                             ).to(device))
                idt = torch.from_numpy(
                    ids[row:row + 1, :gm].astype(np.int64)).to(device
                                                               ).clamp_min(0)
                prob = torch.sigmoid(
                    torch.gather(logit, -1, idt)).cpu().numpy()[0]  # [gm,J,K]
                for k in range(gm):
                    c = coord.get((si, int(row), k))
                    if c is None:
                        continue
                    s, absp, _t = c
                    rc16 = recur[row, k]           # [J,K]
                    m = rc16 >= 0
                    jj, kk = np.nonzero(m)
                    if jj.size == 0:
                        continue
                    cols["seq"].append(np.full(jj.size, s, np.int64))
                    cols["pos"].append(np.full(jj.size, absp, np.int32))
                    cols["layer"].append(jj.astype(np.int16))
                    cols["expert"].append(ids[row, k, jj, kk].astype(np.int16))
                    cols["prob"].append(prob[k, jj, kk].astype(np.float32))
                    cols["label"].append(rc16[jj, kk].astype(np.int8))
                    cols["occ_prev"].append(prev[row, k, jj, kk].astype(np.int8))
                    cols["nextdist"].append(nxt[row, k, jj, kk].astype(np.int16))
    out = {n: np.concatenate(v) for n, v in cols.items()}
    np.savez(out_path, **out)
    log(f"dumped {out['seq'].size:,} held-out prediction slots -> {out_path}")


def fit_logistic(X: np.ndarray, y: np.ndarray, steps: int = 400,
                 lr: float = 0.5) -> np.ndarray:
    """Tiny full-batch logistic regression (finding #3's occurrence
    model). Standardize features; return weights incl. bias."""
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    w = np.zeros(Xs.shape[1])
    yb = y.astype(np.float64)
    for _ in range(steps):
        z = Xs @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xs.T @ (p - yb) / len(yb)
        w -= lr * grad
    return np.concatenate([mu, sd, w])  # pack stats + weights


def apply_logistic(packed: np.ndarray, X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    mu, sd, w = packed[:d], packed[d:2 * d], packed[2 * d:]
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-(Xs @ w)))


def evaluate(mdl, cfg, index, train_keys, held_keys, device, log,
             sub_stride: int, eval_max_shards: int | None = None,
             logistic_shards: int = 8):
    log("collecting held-out eval slots ...")
    ev = collect_eval(mdl, cfg, index, held_keys, device, log, sub_stride,
                      max_shards=eval_max_shards)
    n = ev["label"].size
    base = float((ev["label"] > 0).mean())
    log(f"held-out slots: {n:,}  base positive rate {base:.4f}")

    # fit occurrence logistic on a TRAIN subsample (few shards suffice)
    log("fitting occurrence logistic on train subsample ...")
    tr = collect_eval(mdl, cfg, index, train_keys, device, log,
                      max(sub_stride, 8), max_shards=logistic_shards)
    packed = fit_logistic(tr["occ_feats"], (tr["label"] > 0).astype(np.int64))
    ev_occ_log = apply_logistic(packed, ev["occ_feats"])

    methods = {
        "base_rate": np.zeros(n),
        "occ_prev": ev["occ_prev"],
        "occ_trail": ev["occ_trail"],
        "lru_recency": ev["lru_recency"],
        "occ_logistic": ev_occ_log,
        "draft_hidden": ev["draft_hidden"],
    }
    y = ev["label"]
    table = {}
    for name, sc in methods.items():
        a = auc(sc, y)
        ap = average_precision(sc, y)
        rec, prec = recall_prec_at_budget(sc, y, base)
        table[name] = {"auc": a, "ap": ap, "recall@budget": rec,
                       "precision@budget": prec}
        log(f"  {name:14s} AUC {a:.4f}  AP {ap:.4f}  "
            f"R@budget {rec:.4f}  P@budget {prec:.4f}")

    # per-position (gamma k) and per-layer AUC for the two headline methods
    per_pos = {}
    for name in ("draft_hidden", "occ_prev", "lru_recency"):
        rows = []
        for k in range(int(ev["kpos"].max()) + 1):
            m = ev["kpos"] == k
            if m.sum() < 100:
                continue
            rows.append({"k": int(k), "n": int(m.sum()),
                         "auc": auc(methods[name][m], y[m])})
        per_pos[name] = rows
    per_layer = {}
    for name in ("draft_hidden", "occ_prev", "lru_recency"):
        rows = []
        for j in range(len(index["moe_layers"])):
            m = ev["layer"] == j
            if m.sum() < 100:
                continue
            rows.append({"layer": int(index["moe_layers"][j]),
                         "n": int(m.sum()),
                         "auc": auc(methods[name][m], y[m])})
        per_layer[name] = rows
    return {"n_slots": int(n), "base_rate": base, "overall": table,
            "per_position": per_pos, "per_layer": per_layer,
            "logistic_packed": packed.tolist()}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="build/recur/default")
    ap.add_argument("--routers", required=True,
                    help="existing routers.npz (RouterBank replicas)")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--eval-stride", type=int, default=4)
    ap.add_argument("--eval-max-shards", type=int, default=0,
                    help="cap held-out shards visited at eval (0 = all)")
    ap.add_argument("--logistic-shards", type=int, default=8)
    ap.add_argument("--dump-predictions", default=None,
                    help="also write full held-out predictions npz here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--evaluate-only", default=None,
                    help="checkpoint prefix to eval without training")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    bank = glm_router.RouterBank.from_npz(args.routers)
    if args.evaluate_only:
        index = ds.load_index(Path(cfg["data"]["shards"]))
        mdl, _ = M.load_model(args.evaluate_only, bank, args.device)
        seq_keys = sorted({int(x) for sh in index["shards"]
                           for x in sh["seq_keys"]})
        train_keys, held_keys = ds.sequence_split(
            seq_keys, cfg["data"]["held_out_fraction"],
            cfg["data"]["split_seed"])
    else:
        mdl, index, train_keys, held_keys = train(
            cfg, out, args.device, args.max_steps, log, bank)
    res = evaluate(mdl, cfg, index, train_keys, held_keys, args.device,
                   log, args.eval_stride,
                   eval_max_shards=(args.eval_max_shards or None),
                   logistic_shards=args.logistic_shards)
    json.dump(res, open(out / "recur_eval.json", "w"), indent=1)
    log("wrote", out / "recur_eval.json")
    if args.dump_predictions:
        dump_predictions(mdl, cfg, index, held_keys, args.device,
                         Path(args.dump_predictions), log)


if __name__ == "__main__":
    main()
