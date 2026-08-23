"""P11.b driver: probe-driven model selection on the RESIDUAL target
(spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(4); prerequisite: Stage-0
CONTINUE verdict, studies/epm/p11-stage0/).

Steps (state carried in --workdir):

  dump     Online-causal candidate feature rows (stageb_features) over
           BOTH splits, position-strided; the transition table
           accumulates train -> held (deployment order). Writes
           features_train.npz / features_held.npz.
  fit      (i) per-feature-class logistic probes ON THE RESIDUAL rows
           (candidates outside prev-top8) — S~=0 => class dead;
           (ii) full-linear; (iii) GBDT vs logistic interaction check
           with a per-sequence bootstrap CI on the AUC difference.
           Metrics reported on full rows AND with boundary positives
           removed (addressable-only — wins on boundary slots are
           noise-fitting, Stage-0 summary).
  eval     Full held pass scoring the fitted probes online:
           pool coverage of prev8 ∪ probe-top-k, addressable-mass
           recovery, per-chunk manifest top-up (prev-union ∪ probe
           top-D) vs FREE alternatives (trail64-pool, two-chunk union)
           at matched bytes, evict dumps (protect = prev8 ∪ top-48).
  evict    Process-parallel sims over the dumps (stage0.run_evict).
  verdict  §3-P11(4) gates: dead classes, GBDT>logistic beyond noise,
           and the pre-registered capacity-candidate entry decision
           (both gates must fire). Files the study dir.

The probes are ranking models; scores are consumed as top-M pools
(§3-P11(2)(a)) — boundary churn stops mattering by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import stage0, stageb_features, train_recurrence  # noqa: E402
from elb_train.stageb_features import CLASSES, FEATURES, F_IDX  # noqa: E402

EVICT_PROBE_TOPK = 48        # protect = prev8 ∪ top-48 => budget 56
POOL_KS = (8, 16, 24, 48)    # residual top-k => pool sizes 16..56
MANIFEST_DS = (8, 16, 32)    # per-layer top-up sizes


# ── shared streaming ─────────────────────────────────────────────────────────

def stream_positions(c: stage0.Corpus, seq_keys, hs, sigma, bias,
                     on_position, on_chunk=None, label_c: float = 1.0):
    """Drive HistoryState over the committed timeline of `seq_keys` in
    run order. Per position t (hs.ready only):
      on_position(key, s, t, tops[J,K], top_w, sel[J,E], margins_fn)
    then hs.update. Per chunk start (index ci>=1), BEFORE any of its
    positions update state:
      on_chunk(key, s, ci, blk, ucur_deep_fn) — ucur computed lazily
    from the chunk's FULL labeled rows."""
    seqs = stage0.gather_split_cells(
        c, seq_keys,
        recur_keys=getattr(on_position, "recur_keys", frozenset()))
    n_done = 0
    t0 = time.time()
    for run_idx, keys in stage0._by_run(seqs):
        v = c.view(run_idx)
        for key in keys:
            s = seqs[key]
            recs = c.lookup_records(run_idx, s["seq_id"], s["pos"])
            tops = np.ascontiguousarray(v.top_ids[recs])
            top_w = np.ascontiguousarray(v.top_w[recs])
            sel = stage0._sel_scores(v.logits(recs), bias)
            chunk = s["chunk"]
            hs.reset_seq()
            for ci, blk in enumerate(s["blocks"]):
                if on_chunk is not None:
                    frecs = c.lookup_records(run_idx, s["seq_id"],
                                             blk["full_pos"])
                    ftops = v.top_ids[frecs]
                    on_chunk(key, s, ci, ftops)
                for t in np.nonzero(chunk == ci)[0]:
                    if hs.ready:
                        on_position(key, s, int(t), tops[t], top_w[t],
                                    sel[t])
                    hs.update(tops[t], top_w[t], sel[t])
            n_done += 1
            if n_done % 20 == 0:
                print(f"[stageb] {n_done}/{len(seqs)} seqs "
                      f"({time.time()-t0:.0f}s)", flush=True)
        c.drop_run(run_idx)


# ── dump ─────────────────────────────────────────────────────────────────────

def _collect_rows(c: stage0.Corpus, hs, sigma, bias, seq_keys,
                  stride: int) -> dict[str, np.ndarray]:
    """Strided candidate feature rows over `seq_keys` (one HistoryState,
    online order)."""
    rows: dict[str, list] = {k: [] for k in
                             ("X", "y", "margin", "bnd", "w",
                              "seq", "pos", "layer", "expert")}
    counter = [0]

    def on_position(key, s, t, tops, top_w, sel):
        counter[0] += 1
        if counter[0] % stride:
            return
        for j in range(c.J):
            cand = hs.candidates(j)
            if not len(cand):
                continue
            X = hs.feature_block(j, cand)
            y, margin, bnd = stageb_features.label_rows(
                cand, tops[j], sel[j], float(sigma[j]))
            wslot = np.zeros(len(cand), np.float32)
            member = {int(e): float(wv) for e, wv in
                      zip(tops[j], top_w[j])}
            for i, e in enumerate(cand):
                if y[i]:
                    wslot[i] = member.get(int(e), 0.0)
            rows["X"].append(X)
            rows["y"].append(y)
            rows["margin"].append(margin)
            rows["bnd"].append(bnd)
            rows["w"].append(wslot)
            n = len(cand)
            rows["seq"].append(np.full(n, key, np.int64))
            rows["pos"].append(np.full(n, s["pos"][t], np.int32))
            rows["layer"].append(np.full(n, j, np.int16))
            rows["expert"].append(cand.astype(np.int16))

    stream_positions(c, seq_keys, hs, sigma, bias, on_position)
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v
            in rows.items()}


def _partition(keys, jobs: int) -> list[list[int]]:
    """Round-robin by run (seq_key >> 48) for memmap locality."""
    groups: list[list[int]] = [[] for _ in range(jobs)]
    by_run: dict[int, list[int]] = {}
    for k in keys:
        by_run.setdefault(int(k) >> 48, []).append(int(k))
    for i, run in enumerate(sorted(by_run)):
        groups[i % jobs].extend(sorted(by_run[run]))
    return [g for g in groups if g]


def _dump_worker(args) -> tuple[str, int, int]:
    (shards, routing, seed, frac, seq_keys, sigma_path, bank_path,
     stride, out_path) = args
    c = stage0.Corpus(shards, routing, seed, frac)
    sigma = np.asarray(json.load(open(sigma_path))["sigma"], np.float32)
    from elb_train import glm_router
    bank = glm_router.RouterBank.from_npz(bank_path)
    hs = stageb_features.HistoryState(c.J, c.E)
    arrs = _collect_rows(c, hs, sigma, bank.bias, seq_keys, stride)
    np.savez_compressed(out_path, **arrs)
    return out_path, int(len(arrs["y"])), int(arrs["y"].sum())


def step_dump(c: stage0.Corpus, sigma, bias, work: Path,
              stride: int, jobs: int = 0, sigma_path: str = "",
              shards: str = "", routing: str = "", seed: int = 0,
              frac: float = 0.25) -> dict:
    info = {}
    if jobs <= 1:
        hs = stageb_features.HistoryState(c.J, c.E)
        for split_name, keys in (("train", c.train_keys),
                                 ("held", c.held_keys)):
            arrs = _collect_rows(c, hs, sigma, bias, keys, stride)
            out = work / f"features_{split_name}.npz"
            np.savez_compressed(out, **arrs)
            info[split_name] = {"rows": int(len(arrs["y"])),
                                "positives": int(arrs["y"].sum()),
                                "path": str(out)}
            print(f"[stageb] dump {split_name}: {info[split_name]}",
                  flush=True)
        return info
    # sequence-parallel: per-worker cold HistoryState (Stage-0 measured
    # the only cross-sequence state, the transition table, warm == cold).
    import concurrent.futures as cf
    bank_path = str(work / "router_bank.npz")
    for split_name, keys in (("train", c.train_keys),
                             ("held", c.held_keys)):
        groups = _partition(keys, jobs)
        tasks = [(shards, routing, seed, frac, g, sigma_path, bank_path,
                  stride, str(work / f"features_{split_name}_p{i}.npz"))
                 for i, g in enumerate(groups)]
        parts = []
        # spawn context: NEVER fork after OpenMP (lightgbm) has run in
        # this process — forked children inherit a locked OMP runtime
        # and deadlock on their first parallel region.
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=mp.get_context("spawn")) as ex:
            for path, n, pos in ex.map(_dump_worker, tasks):
                parts.append(path)
        merged = {}
        for k in ("X", "y", "margin", "bnd", "w", "seq", "pos",
                  "layer", "expert"):
            merged[k] = np.concatenate(
                [np.load(p)[k] for p in parts])
        out = work / f"features_{split_name}.npz"
        np.savez_compressed(out, **merged)
        for p in parts:
            Path(p).unlink()
        info[split_name] = {"rows": int(len(merged["y"])),
                            "positives": int(merged["y"].sum()),
                            "path": str(out), "jobs": len(tasks)}
        print(f"[stageb] dump {split_name}: {info[split_name]}",
              flush=True)
    return info


# ── fit ──────────────────────────────────────────────────────────────────────

def _residual_mask(d: dict) -> np.ndarray:
    """The residual task: candidates NOT already predicted by b0_prev."""
    return d["X"][:, F_IDX["in_prev"]] == 0


class _WeightedAuc:
    """One-sort weighted Mann-Whitney AUC: precompute the score order and
    tie groups once; each evaluation with per-row weights is two gathers
    + one reduceat (a per-sequence bootstrap resample = integer row
    weights, so 200 resamples cost 200 vectorized passes, not 200
    argsorts of the full row set)."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        self.order = np.argsort(score, kind="stable")
        s = score[self.order]
        starts = np.nonzero(np.r_[True, s[1:] != s[:-1]])[0]
        self.starts = starts
        self.y_sorted = y[self.order].astype(np.float64)

    def auc(self, w: np.ndarray) -> float:
        ws = w[self.order].astype(np.float64)
        wpos = ws * self.y_sorted
        wneg = ws - wpos
        gpos = np.add.reduceat(wpos, self.starts)
        gneg = np.add.reduceat(wneg, self.starts)
        cneg = np.cumsum(gneg) - gneg          # negatives strictly below
        U = float((gpos * (cneg + 0.5 * gneg)).sum())
        tp, tn = float(wpos.sum()), float(wneg.sum())
        return U / (tp * tn) if tp > 0 and tn > 0 else 0.5


def _auc_boot_diff(sa: np.ndarray, sb: np.ndarray, y: np.ndarray,
                   seq: np.ndarray, n_boot: int = 200,
                   seed: int = 0) -> dict:
    """Per-sequence bootstrap CI of AUC(sa) - AUC(sb)."""
    rng = np.random.default_rng(seed)
    uniq, seq_idx = np.unique(seq, return_inverse=True)
    wa = _WeightedAuc(np.asarray(sa, np.float64), y)
    wb = _WeightedAuc(np.asarray(sb, np.float64), y)
    diffs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)),
                             minlength=len(uniq))
        w = counts[seq_idx].astype(np.float64)
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        diffs.append(wa.auc(w) - wb.auc(w))
    d = np.array(diffs)
    return {"mean": float(d.mean()), "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)), "n_boot": len(d)}


def step_fit(work: Path, fit_cap: int, gbm_rounds: int) -> dict:
    tr = dict(np.load(work / "features_train.npz"))
    he = dict(np.load(work / "features_held.npz"))
    out: dict = {"probes": {}}
    rtr = _residual_mask(tr)
    rhe = _residual_mask(he)
    Xtr, ytr = tr["X"][rtr], tr["y"][rtr]
    Xhe, yhe = he["X"][rhe], he["y"][rhe]
    seq_he = he["seq"][rhe]
    addr = ~(he["bnd"][rhe] > 0)      # drop boundary POSITIVES for strat
    rng = np.random.default_rng(0)
    if len(ytr) > fit_cap:
        pick = rng.choice(len(ytr), fit_cap, replace=False)
        Xfit, yfit = Xtr[pick], ytr[pick]
    else:
        Xfit, yfit = Xtr, ytr
    print(f"[stageb] fit rows: train {len(ytr):,} (fit on "
          f"{len(yfit):,}), held {len(yhe):,} "
          f"(pos rate {yhe.mean():.4f})", flush=True)

    def bench(name, score_he, model=None):
        m = {
            "auc": float(train_recurrence.auc(score_he, yhe)),
            "ap": float(train_recurrence.average_precision(score_he,
                                                           yhe)),
            "auc_addressable": float(train_recurrence.auc(
                score_he[addr], yhe[addr])),
        }
        out["probes"][name] = m
        print(f"[stageb] {name}: auc {m['auc']:.4f} "
              f"(addressable {m['auc_addressable']:.4f})", flush=True)
        return m

    # per-class + full logistic probes on the residual target
    scores = {}
    for cls, feats in list(CLASSES.items()) + [("full", FEATURES)]:
        cols = [F_IDX[f] for f in feats]
        packed = train_recurrence.fit_logistic(
            Xfit[:, cols], yfit.astype(np.float32))
        sc = train_recurrence.apply_logistic(packed, Xhe[:, cols])
        scores[cls] = sc
        bench(f"logistic_{cls}", sc)
        np.save(work / f"probe_{cls}.npy", packed)
        out["probes"][f"logistic_{cls}"]["features"] = list(feats)

    # GBDT interaction check (fusion.py precedent)
    import lightgbm as lgb
    dtr = lgb.Dataset(Xfit, label=yfit)
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "num_leaves": 63, "learning_rate": 0.08,
              "feature_fraction": 0.9, "seed": 0}
    gbm = lgb.train(params, dtr, num_boost_round=gbm_rounds)
    sc_gbm = gbm.predict(Xhe, num_iteration=gbm.best_iteration)
    bench("gbdt", sc_gbm)
    gbm.save_model(str(work / "probe_gbdt.txt"))
    out["gbdt_importance_gain"] = dict(
        zip(FEATURES, [float(x) for x in
                       gbm.feature_importance("gain")]))
    out["gbdt_vs_logistic"] = _auc_boot_diff(
        np.asarray(sc_gbm, np.float64),
        np.asarray(scores["full"], np.float64), yhe, seq_he)
    out["gbdt_vs_logistic_addressable"] = _auc_boot_diff(
        np.asarray(sc_gbm, np.float64)[addr],
        np.asarray(scores["full"], np.float64)[addr], yhe[addr],
        seq_he[addr])
    print(f"[stageb] gbdt-vs-logistic AUC diff "
          f"{out['gbdt_vs_logistic']['mean']:+.4f} "
          f"[{out['gbdt_vs_logistic']['lo95']:+.4f}, "
          f"{out['gbdt_vs_logistic']['hi95']:+.4f}]", flush=True)
    return out


# ── eval (deployed proxies) ──────────────────────────────────────────────────

class _LinearScorer:
    def __init__(self, work: Path):
        self.packed = np.load(work / "probe_full.npy")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return train_recurrence.apply_logistic(self.packed, X)


class _GbdtScorer:
    def __init__(self, work: Path, num_threads: int = 4):
        import lightgbm as lgb
        self.gbm = lgb.Booster(model_file=str(work / "probe_gbdt.txt"))
        # cap OMP threads per predict so 24 workers don't oversubscribe.
        # NOTE: passed as a pred_parameter — Booster.reset_parameter()
        # SEGFAULTS on a file-loaded predictor-only booster (lgb 4.7).
        self.num_threads = num_threads

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.gbm.predict(
            X, num_threads=self.num_threads).astype(np.float32)


def _eval_seqs(work: Path, c: stage0.Corpus, sigma, bias, held_keys,
               evict_keys: set[int], deep_lo: int, deep_hi: int,
               warm_keys, dump_suffix: str = "") -> dict:
    """Deployed-proxy accumulation over `held_keys` (one HistoryState,
    warmed over `warm_keys` first). Returns a MERGEABLE partial; evict
    dumps are written with `dump_suffix` for parent-side concatenation."""
    scorers = {"linear": _LinearScorer(work), "gbdt": _GbdtScorer(work)}
    deep_idx = np.array([i for i, l in enumerate(c.moe_layers)
                         if deep_lo <= l <= deep_hi])
    hs = stageb_features.HistoryState(c.J, c.E)
    if len(warm_keys):
        print("[stageb] warming state through the train split ...",
              flush=True)
        stream_positions(c, warm_keys, hs, sigma, bias, lambda *a: None)
    held = sorted(int(s) for s in held_keys)

    # accumulators
    pool_hit = {(nm, k): 0 for nm in scorers for k in POOL_KS}
    pool_true = [0]
    addr_hit = {(nm, k): 0 for nm in scorers for k in POOL_KS}
    addr_n = [0]
    man = {(nm, d): [0.0, 0] for nm in scorers for d in MANIFEST_DS}
    man_free = {alt: [0.0, 0.0, 0] for alt in
                ("prev_union", "two_chunk", "trail64_pool")}
    dump: dict[str, dict[str, list]] = {
        nm: {k: [] for k in ("seq", "pos", "layer", "expert", "prob",
                             "label", "occ_prev", "nextdist")}
        for nm in scorers}
    prev_ucur: dict[int, list] = {}

    def on_chunk(key, s, ci, ftops):
        # U_cur over full rows, deep layers
        ucur = np.zeros((len(deep_idx), c.E), bool)
        ft = ftops[:, deep_idx].astype(np.int64)
        for row in ft:
            np.put_along_axis(ucur, row, True, axis=1)
        state = prev_ucur.setdefault(key, [None, None])
        u1, u2 = state[0], state[1]
        if u1 is not None and hs.ready:
            un = ucur.sum(axis=1)
            ok = un > 0
            nok = int(ok.sum())
            if nok:
                # free baselines
                cov1 = ((u1 & ucur).sum(axis=1)[ok] / un[ok]).sum()
                man_free["prev_union"][0] += cov1
                man_free["prev_union"][1] += float(u1[ok].sum())
                man_free["prev_union"][2] += nok
                u12 = u1 | u2 if u2 is not None else u1
                cov2 = ((u12 & ucur).sum(axis=1)[ok] / un[ok]).sum()
                man_free["two_chunk"][0] += cov2
                man_free["two_chunk"][1] += float(u12[ok].sum())
                man_free["two_chunk"][2] += nok
                t64 = hs.trail64.counts[deep_idx] > 0
                cov3 = ((t64 & ucur).sum(axis=1)[ok] / un[ok]).sum()
                man_free["trail64_pool"][0] += cov3
                man_free["trail64_pool"][1] += float(t64[ok].sum())
                man_free["trail64_pool"][2] += nok
                # probe top-up: prev-union ∪ top-D non-union candidates
                # (one batched predict per scorer across deep layers)
                layer_blocks = []
                Xs = []
                off = 0
                for di, jj in enumerate(deep_idx):
                    if un[di] == 0:
                        continue
                    cand = hs.candidates(int(jj))
                    if not len(cand):
                        continue
                    Xs.append(hs.feature_block(int(jj), cand))
                    layer_blocks.append(
                        (di, cand, slice(off, off + len(cand))))
                    off += len(cand)
                if layer_blocks:
                    Xall = np.concatenate(Xs)
                    for nm, fn in scorers.items():
                        sc_all = fn(Xall)
                        for di, cand, sl in layer_blocks:
                            sc = sc_all[sl]
                            outside = np.nonzero(~u1[di][cand])[0]
                            order = outside[np.argsort(-sc[outside])]
                            for d in MANIFEST_DS:
                                m = u1[di].copy()
                                m[cand[order[:d]]] = True
                                cov = (m & ucur[di]).sum() / un[di]
                                man[(nm, d)][0] += float(cov)
                                man[(nm, d)][1] += 1
        state[1] = state[0]
        state[0] = ucur

    def on_position(key, s, t, tops, top_w, sel):
        is_evict = key in evict_keys and s["recur"] is not None
        layer_blocks = []
        Xs = []
        off = 0
        for j in range(c.J):
            cand = hs.candidates(j)
            if not len(cand):
                continue
            X = hs.feature_block(j, cand)
            y, _margin, bnd = stageb_features.label_rows(
                cand, tops[j], sel[j], float(sigma[j]))
            in_prev = X[:, F_IDX["in_prev"]] > 0
            addressable = (y > 0) & ~in_prev & ~(bnd > 0)
            layer_blocks.append((j, cand, y > 0, in_prev, addressable,
                                 slice(off, off + len(cand))))
            Xs.append(X)
            off += len(cand)
            pool_true[0] += c.K
            addr_n[0] += int(addressable.sum())
        if not layer_blocks:
            return
        Xall = np.concatenate(Xs)
        for nm, fn in scorers.items():
            sc_all = fn(Xall)
            for j, cand, yb, in_prev, addressable, sl in layer_blocks:
                sc = sc_all[sl]
                resid = np.nonzero(~in_prev)[0]
                order = resid[np.argsort(-sc[resid])]
                prev_hits = int((yb & in_prev).sum())
                cum_y = np.cumsum(yb[order])
                cum_a = np.cumsum(addressable[order])
                m = len(order)
                for k in POOL_KS:
                    kk = min(k, m)
                    pool_hit[(nm, k)] += prev_hits + (
                        int(cum_y[kk - 1]) if kk else 0)
                    addr_hit[(nm, k)] += int(cum_a[kk - 1]) if kk else 0
                if is_evict:
                    prot = np.zeros(c.E, bool)
                    prot[cand[order[:EVICT_PROBE_TOPK]]] = True
                    prot[cand[in_prev]] = True
                    dm = dump[nm]
                    t8 = tops[j].astype(np.int64)
                    dm["seq"].append(np.full(c.K, key, np.int64))
                    dm["pos"].append(np.full(c.K, s["pos"][t], np.int32))
                    dm["layer"].append(np.full(c.K, j, np.int16))
                    dm["expert"].append(tops[j].astype(np.int16))
                    dm["prob"].append(prot[t8].astype(np.float32))
                    dm["label"].append(s["recur"]["label"][t, j])
                    dm["occ_prev"].append(s["recur"]["occ_prev"][t, j])
                    dm["nextdist"].append(s["recur"]["nextdist"][t, j])

    on_position.recur_keys = evict_keys
    print(f"[stageb] deployed eval over {len(held)} held seqs ...",
          flush=True)
    stream_positions(c, held, hs, sigma, bias, on_position,
                     on_chunk=on_chunk)

    dump_paths: dict[str, str] = {}
    for nm, dm in dump.items():
        if dm["seq"]:
            arrs = {k: np.concatenate(v) for k, v in dm.items()}
            p = work / f"preds_probe_{nm}{dump_suffix}.npz"
            np.savez_compressed(p, **arrs)
            dump_paths[nm] = str(p)
    return {
        "pool_hit": {f"{nm}|{k}": v for (nm, k), v in pool_hit.items()},
        "pool_true": pool_true[0],
        "addr_hit": {f"{nm}|{k}": v for (nm, k), v in addr_hit.items()},
        "addr_n": addr_n[0],
        "man": {f"{nm}|{d}": v for (nm, d), v in man.items()},
        "man_free": man_free,
        "dump_paths": dump_paths,
        "scorers": list(scorers),
    }


def _eval_worker(args) -> dict:
    (shards, routing, seed, frac, held_group, evict_list, sigma_path,
     workdir, deep_lo, deep_hi, wid) = args
    c = stage0.Corpus(shards, routing, seed, frac)
    sigma = np.asarray(json.load(open(sigma_path))["sigma"], np.float32)
    from elb_train import glm_router
    bank = glm_router.RouterBank.from_npz(
        Path(workdir) / "router_bank.npz")
    return _eval_seqs(Path(workdir), c, sigma, bank.bias, held_group,
                      set(evict_list), deep_lo, deep_hi, c.train_keys,
                      dump_suffix=f"_p{wid}")


def step_eval(c: stage0.Corpus, sigma, bias, work: Path,
              evict_max_seqs: int, deep_lo: int, deep_hi: int,
              jobs: int = 0, sigma_path: str = "", shards: str = "",
              routing: str = "", seed: int = 0,
              frac: float = 0.25) -> dict:
    held = sorted(int(s) for s in c.held_keys)
    evict_keys = set(held[:evict_max_seqs])
    if jobs <= 1:
        partials = [_eval_seqs(work, c, sigma, bias, held, evict_keys,
                               deep_lo, deep_hi, c.train_keys)]
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        groups = _partition(held, jobs)
        tasks = [(shards, routing, seed, frac, g,
                  sorted(evict_keys & set(g)), sigma_path, str(work),
                  deep_lo, deep_hi, i) for i, g in enumerate(groups)]
        # spawn: see step_dump — fork-after-OpenMP deadlocks.
        with cf.ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=mp.get_context("spawn")) as ex:
            partials = list(ex.map(_eval_worker, tasks))

    # merge partials
    scorers = partials[0]["scorers"]
    pool_hit = {k: sum(p["pool_hit"][k] for p in partials)
                for k in partials[0]["pool_hit"]}
    addr_hit = {k: sum(p["addr_hit"][k] for p in partials)
                for k in partials[0]["addr_hit"]}
    pool_true = sum(p["pool_true"] for p in partials)
    addr_n = sum(p["addr_n"] for p in partials)
    man = {k: [sum(p["man"][k][0] for p in partials),
               sum(p["man"][k][1] for p in partials)]
           for k in partials[0]["man"]}
    man_free = {alt: [sum(p["man_free"][alt][i] for p in partials)
                      for i in range(3)]
                for alt in partials[0]["man_free"]}

    out: dict = {"pool": {}, "manifest": {}, "evict_dumps": {}}
    for nm in scorers:
        out["pool"][nm] = {
            f"k{k}": {
                "coverage": pool_hit[f"{nm}|{k}"] / max(1, pool_true),
                "pool_size": c.K + k,
                "addressable_recovery":
                    addr_hit[f"{nm}|{k}"] / max(1, addr_n),
            } for k in POOL_KS}
    for alt, (cov, size, n) in man_free.items():
        out["manifest"][alt] = {"cov_deep": cov / max(1, n),
                                "avg_size": size / max(1, n)}
    base_size = out["manifest"]["prev_union"]["avg_size"]
    for nm in scorers:
        for d in MANIFEST_DS:
            cov, n = man[f"{nm}|{d}"]
            out["manifest"][f"{nm}_topup{d}"] = {
                "cov_deep": cov / max(1, n),
                "avg_size": base_size + d}
    for nm in scorers:
        parts = [p["dump_paths"][nm] for p in partials
                 if nm in p["dump_paths"]]
        if parts:
            merged = {k: np.concatenate([np.load(pp)[k] for pp in parts])
                      for k in ("seq", "pos", "layer", "expert", "prob",
                                "label", "occ_prev", "nextdist")}
            p = work / f"preds_probe_{nm}.npz"
            np.savez_compressed(p, **merged)
            out["evict_dumps"][f"probe_{nm}"] = str(p)
            for pp in parts:
                if pp != str(p):
                    Path(pp).unlink()
    out["addressable_slots"] = addr_n
    out["jobs"] = max(1, jobs)
    return out


# ── verdict ──────────────────────────────────────────────────────────────────

def evaluate_stageb_verdict(fit: dict, ev: dict, evict: dict | None,
                            dead_class_bar: float = 0.52,
                            addr_recovery_bar: float = 0.5) -> dict:
    """§3-P11(4) gate logic. Thresholds chosen HERE (stated, not
    pre-registered): a class is dead when its residual AUC < 0.52
    (chance + noise floor); 'linear leaves residual skill' when
    linear addressable recovery at k=8 is under `addr_recovery_bar`
    of the k=48 ceiling; GBDT gate fires when the bootstrap 95% CI of
    the AUC difference excludes 0 above it."""
    probes = fit["probes"]
    classes = {}
    for cls in list(CLASSES) + ["full"]:
        m = probes[f"logistic_{cls}"]
        classes[cls] = {"auc": m["auc"],
                        "auc_addressable": m["auc_addressable"],
                        "dead": m["auc"] < dead_class_bar}
    ci = fit["gbdt_vs_logistic"]
    gbdt_gate = ci["lo95"] > 0.0
    lin = ev["pool"]["linear"]
    rec8 = lin[f"k{POOL_KS[0]}"]["addressable_recovery"]
    rec_max = lin[f"k{POOL_KS[-1]}"]["addressable_recovery"]
    linear_leaves_skill = rec_max > 0 and (rec8 / rec_max) < addr_recovery_bar
    capacity_enters = bool(gbdt_gate and linear_leaves_skill)
    return {
        "classes": classes,
        "gbdt_vs_logistic_ci": ci,
        "gbdt_gate_fires": bool(gbdt_gate),
        "linear_addr_recovery_k8": rec8,
        "linear_addr_recovery_kmax": rec_max,
        "linear_leaves_residual_skill": bool(linear_leaves_skill),
        "capacity_candidate_enters": capacity_enters,
        "bars": {"dead_class_auc": dead_class_bar,
                 "addr_recovery_ratio": addr_recovery_bar},
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--router-ckpt", default=stage0.DEFAULT_ROUTER_CKPT)
    ap.add_argument("--split-seed", type=int, default=35)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--do", default="dump,fit,eval,evict,verdict")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=0,
                    help="sequence-parallel workers for dump/eval "
                         "(0 = serial; per-worker transition tables — "
                         "Stage-0 measured warm==cold)")
    ap.add_argument("--fit-cap", type=int, default=8_000_000)
    ap.add_argument("--gbm-rounds", type=int, default=400)
    ap.add_argument("--sigma", default="build/recur/p11-stage0/sigma.json")
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--evict-max-seqs", type=int, default=24)
    ap.add_argument("--evict-caps", default="700,1052")
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--workdir", default="build/recur/p11-stageb")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.do.split(",") if s.strip()]
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    c = stage0.Corpus(a.shards, a.routing, a.split_seed, a.held_fraction)
    if a.limit_seqs:
        c.train_keys = c.train_keys[: max(2, a.limit_seqs * 3)]
        c.held_keys = c.held_keys[: a.limit_seqs]
    sigma = np.asarray(json.load(open(a.sigma))["sigma"], np.float32)
    bank_npz = work / "router_bank.npz"
    from elb_train import glm_router
    if bank_npz.is_file():
        bank = glm_router.RouterBank.from_npz(bank_npz)
    else:
        src = Path("build/recur/p11-stage0/router_bank.npz")
        bank = (glm_router.RouterBank.from_npz(src) if src.is_file()
                else glm_router.RouterBank.from_checkpoint(
                    a.router_ckpt, c.moe_layers))
        bank.save_npz(bank_npz)

    # merge into any prior results so per-step invocations (the e2e
    # pipeline) accumulate instead of clobbering earlier steps.
    rpath = work / "results.json"
    results: dict = json.load(open(rpath)) if rpath.is_file() else {}
    results["config"] = vars(a)

    def _checkpoint():
        json.dump(results, open(rpath, "w"), indent=1)

    if "dump" in steps:
        print("[stageb] dump ...", flush=True)
        results["dump"] = step_dump(
            c, sigma, bank.bias, work, a.stride, jobs=a.jobs,
            sigma_path=a.sigma, shards=a.shards, routing=a.routing,
            seed=a.split_seed, frac=a.held_fraction)
        _checkpoint()
    if "fit" in steps:
        print("[stageb] fit ...", flush=True)
        results["fit"] = step_fit(work, a.fit_cap, a.gbm_rounds)
        _checkpoint()
    if "eval" in steps:
        print("[stageb] eval ...", flush=True)
        results["eval"] = step_eval(
            c, sigma, bank.bias, work, a.evict_max_seqs, a.deep_lo,
            a.deep_hi, jobs=a.jobs, sigma_path=a.sigma,
            shards=a.shards, routing=a.routing, seed=a.split_seed,
            frac=a.held_fraction)
        _checkpoint()
    if "evict" in steps:
        dumps = (results.get("eval") or {}).get("evict_dumps") or {
            n: str(work / f"preds_{n}.npz")
            for n in ("probe_linear", "probe_gbdt")
            if (work / f"preds_{n}.npz").is_file()}
        if dumps:
            caps = [int(x) for x in a.evict_caps.split(",")]
            results["evict"] = stage0.run_evict(dumps, caps)
    if "verdict" in steps:
        fit = results.get("fit") or json.load(
            open(work / "results.json"))["fit"]
        ev = results.get("eval") or json.load(
            open(work / "results.json"))["eval"]
        results["verdict"] = evaluate_stageb_verdict(
            fit, ev, results.get("evict"))
        print(json.dumps(results["verdict"], indent=1), flush=True)

    json.dump(results, open(work / "results.json", "w"), indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
    print(f"[stageb] wrote {work / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
