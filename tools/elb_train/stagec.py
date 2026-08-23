"""P11.c driver: the online tiny model — per-layer Hedge mixture over the
free-signal bank + a frozen linear probe + an ONLINE ridge learner
(spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(5); prereqs: Stage-0
CONTINUE + P11.b gates, studies/epm/p11-stage{0,b}).

Members combined per (position, layer) on the trail-64 candidate set,
each rank-normalized to [0, 1]:

  b0_prev, trans_ema, score_ema2, trail16   — the free bank (HistoryState)
  probe_frozen                              — the P11.b fitted linear probe
  rls_online                                — decayed-sufficient-statistics
        ridge on the SAME 12 features, closed-form re-solve every
        `refresh` positions (no backprop; labels are free at every
        position — the never-regime-stale learner)

Hedge (multiplicative weights, Freund-Schapire): per-layer weights
w_i <- w_i * exp(-eta * loss_i), loss_i = 1 - recall@8 of member i's
candidate top-8 at that position; regret <= sqrt(T ln N). Kilobytes of
state, microseconds of update — the deployed cost class of §3-P11(1).

Deployed metrics mirror stageb exactly (pools / manifest top-up / evict
protocol) so the mixture, the frozen probe, and the free bank are judged
on the same pass. The mixture's deployed numbers become THE BAR the
admitted §3-P11(4)(iii) capacity candidate must beat at matched compute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import stage0, stageb, stageb_features  # noqa: E402
from elb_train.stageb import (EVICT_PROBE_TOPK, MANIFEST_DS,  # noqa: E402
                              POOL_KS, _partition)
from elb_train.stageb_features import F_IDX, FEATURES  # noqa: E402

BANK = ("b0_prev", "trans_ema", "score_ema2", "trail16",
        "probe_frozen", "rls_online")
HEDGE_ETA = 0.1
RLS_DECAY = 0.999
RLS_LAMBDA = 1.0
RLS_REFRESH = 64


class OnlineRidge:
    """Per-layer decayed-sufficient-statistics ridge on the stageb
    features: A_j = sum decay^age x x^T, b_j = sum decay^age x y; w_j
    re-solved every `refresh` positions (closed-form, no backprop)."""

    def __init__(self, n_layers: int, d: int = len(FEATURES) + 1,
                 decay: float = RLS_DECAY, lam: float = RLS_LAMBDA,
                 refresh: int = RLS_REFRESH):
        self.J, self.d = n_layers, d
        self.decay, self.lam, self.refresh = decay, lam, refresh
        self.A = np.zeros((n_layers, d, d), np.float64)
        self.b = np.zeros((n_layers, d), np.float64)
        self.w = np.zeros((n_layers, d), np.float64)
        self._since = 0

    def _aug(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [X.astype(np.float64), np.ones((len(X), 1))], axis=1)

    def predict(self, j: int, X: np.ndarray) -> np.ndarray:
        return self._aug(X) @ self.w[j]

    def update(self, j: int, X: np.ndarray, y: np.ndarray) -> None:
        Xa = self._aug(X)
        self.A[j] = self.A[j] * self.decay + Xa.T @ Xa
        self.b[j] = self.b[j] * self.decay + Xa.T @ y.astype(np.float64)

    def tick(self) -> None:
        """Once per POSITION (after all layers updated)."""
        self._since += 1
        if self._since >= self.refresh:
            eye = np.eye(self.d) * self.lam
            for j in range(self.J):
                self.w[j] = np.linalg.solve(self.A[j] + eye, self.b[j])
            self._since = 0


class HedgeMixer:
    """Per-layer multiplicative weights over the members."""

    def __init__(self, n_layers: int, n_members: int,
                 eta: float = HEDGE_ETA):
        self.eta = eta
        self.w = np.full((n_layers, n_members), 1.0 / n_members)

    def combine(self, j: int, norm_scores: np.ndarray) -> np.ndarray:
        """norm_scores [N, n_cand] rank-normalized -> combined [n_cand]."""
        return self.w[j] @ norm_scores

    def update(self, j: int, losses: np.ndarray) -> None:
        self.w[j] *= np.exp(-self.eta * losses)
        self.w[j] /= self.w[j].sum()


def _rank_norm(sc: np.ndarray) -> np.ndarray:
    n = len(sc)
    if n <= 1:
        return np.zeros(n, np.float64)
    order = np.argsort(np.argsort(sc, kind="stable"), kind="stable")
    return order / (n - 1)


# ── eval pass (warm-online + held metrics) ───────────────────────────────────

def _stagec_eval_seqs(work: Path, c: stage0.Corpus, sigma, bias,
                      held_keys, evict_keys: set[int], deep_lo: int,
                      deep_hi: int, warm_keys,
                      evict_scorer: str = "rls_online") -> dict:
    probe = stageb._LinearScorer(work)
    deep_idx = np.array([i for i, l in enumerate(c.moe_layers)
                         if deep_lo <= l <= deep_hi])
    hs = stageb_features.HistoryState(c.J, c.E)
    ridge = OnlineRidge(c.J)
    hedge = HedgeMixer(c.J, len(BANK))
    collect = {"on": False}     # metrics off during warm

    # metric accumulators — variants judged on the SAME pass
    variants = ("combined", "probe_frozen", "rls_online", "trans_ema")
    pool_hit = {(v, k): 0 for v in variants for k in POOL_KS}
    addr_hit = {(v, k): 0 for v in variants for k in POOL_KS}
    pool_true = [0]
    addr_n = [0]
    man = {d: [0.0, 0] for d in MANIFEST_DS}          # combined top-up
    man_free = {alt: [0.0, 0.0, 0] for alt in ("prev_union", "two_chunk")}
    dump: dict[str, list] = {k: [] for k in
                             ("seq", "pos", "layer", "expert", "prob",
                              "label", "occ_prev", "nextdist")}
    prev_ucur: dict[int, list] = {}

    def member_scores(j: int, cand: np.ndarray, X: np.ndarray
                      ) -> np.ndarray:
        frame = hs._frame()
        raw = [hs.prev8[j, cand],
               frame["trans"][j, cand],
               frame["ema2"][j, cand],
               hs.trail16.counts[j, cand],
               probe(X),
               ridge.predict(j, X)]
        return np.stack([_rank_norm(r) for r in raw])   # [N, n_cand]

    def on_chunk(key, s, ci, ftops):
        if not collect["on"]:
            return
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
                cov1 = ((u1 & ucur).sum(axis=1)[ok] / un[ok]).sum()
                man_free["prev_union"][0] += cov1
                man_free["prev_union"][1] += float(u1[ok].sum())
                man_free["prev_union"][2] += nok
                u12 = u1 | u2 if u2 is not None else u1
                cov2 = ((u12 & ucur).sum(axis=1)[ok] / un[ok]).sum()
                man_free["two_chunk"][0] += cov2
                man_free["two_chunk"][1] += float(u12[ok].sum())
                man_free["two_chunk"][2] += nok
                for di, jj in enumerate(deep_idx):
                    if un[di] == 0:
                        continue
                    cand = hs.candidates(int(jj))
                    if not len(cand):
                        continue
                    X = hs.feature_block(int(jj), cand)
                    sc = hedge.combine(int(jj),
                                       member_scores(int(jj), cand, X))
                    outside = np.nonzero(~u1[di][cand])[0]
                    order = outside[np.argsort(-sc[outside])]
                    for d in MANIFEST_DS:
                        m = u1[di].copy()
                        m[cand[order[:d]]] = True
                        cov = (m & ucur[di]).sum() / un[di]
                        man[d][0] += float(cov)
                        man[d][1] += 1
        state[1] = state[0]
        state[0] = ucur

    def on_position(key, s, t, tops, top_w, sel):
        is_evict = (collect["on"] and key in evict_keys
                    and s["recur"] is not None)
        for j in range(c.J):
            cand = hs.candidates(j)
            if not len(cand):
                continue
            X = hs.feature_block(j, cand)
            y, _m, bnd = stageb_features.label_rows(
                cand, tops[j], sel[j], float(sigma[j]))
            yb = y > 0
            in_prev = X[:, F_IDX["in_prev"]] > 0
            addressable = yb & ~in_prev & ~(bnd > 0)
            norm = member_scores(j, cand, X)              # [N, n_cand]
            comb = hedge.w[j] @ norm
            # ── metrics (held only) ──
            if collect["on"]:
                pool_true[0] += c.K
                addr_n[0] += int(addressable.sum())
                var_scores = {"combined": comb,
                              "probe_frozen": norm[4],
                              "rls_online": norm[5],
                              "trans_ema": norm[1]}
                for v, sc in var_scores.items():
                    resid = np.nonzero(~in_prev)[0]
                    order = resid[np.argsort(-sc[resid])]
                    prev_hits = int((yb & in_prev).sum())
                    cum_y = np.cumsum(yb[order])
                    cum_a = np.cumsum(addressable[order])
                    m = len(order)
                    for k in POOL_KS:
                        kk = min(k, m)
                        pool_hit[(v, k)] += prev_hits + (
                            int(cum_y[kk - 1]) if kk else 0)
                        addr_hit[(v, k)] += (int(cum_a[kk - 1])
                                             if kk else 0)
                    if v == evict_scorer and is_evict:
                        prot = np.zeros(c.E, bool)
                        prot[cand[order[:EVICT_PROBE_TOPK]]] = True
                        prot[cand[in_prev]] = True
                        t8 = tops[j].astype(np.int64)
                        dump["seq"].append(np.full(c.K, key, np.int64))
                        dump["pos"].append(
                            np.full(c.K, s["pos"][t], np.int32))
                        dump["layer"].append(np.full(c.K, j, np.int16))
                        dump["expert"].append(tops[j].astype(np.int16))
                        dump["prob"].append(prot[t8].astype(np.float32))
                        dump["label"].append(s["recur"]["label"][t, j])
                        dump["occ_prev"].append(
                            s["recur"]["occ_prev"][t, j])
                        dump["nextdist"].append(
                            s["recur"]["nextdist"][t, j])
            # ── online updates: hedge losses + ridge labels (free) ──
            tcand = int(yb.sum())
            if tcand:
                losses = np.empty(len(BANK))
                for i in range(len(BANK)):
                    top8 = np.argpartition(
                        -norm[i], min(8, len(cand)) - 1)[:8]
                    losses[i] = 1.0 - yb[top8].sum() / min(8, tcand)
                hedge.update(j, losses)
            ridge.update(j, X, y.astype(np.float64))
        ridge.tick()

    on_position.recur_keys = evict_keys

    # warm-online over the train split (updates on, metrics off)
    if len(warm_keys):
        print(f"[stagec] warm-online over {len(warm_keys)} train seqs ...",
              flush=True)
        stageb.stream_positions(c, warm_keys, hs, sigma, bias,
                                on_position, on_chunk=on_chunk)
    collect["on"] = True
    print(f"[stagec] held eval over {len(held_keys)} seqs ...", flush=True)
    stageb.stream_positions(c, held_keys, hs, sigma, bias,
                            on_position, on_chunk=on_chunk)

    dump_path = None
    if dump["seq"]:
        arrs = {k: np.concatenate(v) for k, v in dump.items()}
        p = work / "preds_stagec_combined.npz"
        np.savez_compressed(p, **arrs)
        dump_path = str(p)
    ml = c.moe_layers
    deep_mask = (ml >= deep_lo) & (ml <= deep_hi)
    return {
        "pool_hit": {f"{v}|{k}": n for (v, k), n in pool_hit.items()},
        "addr_hit": {f"{v}|{k}": n for (v, k), n in addr_hit.items()},
        "pool_true": pool_true[0], "addr_n": addr_n[0],
        "man": {str(d): v for d, v in man.items()},
        "man_free": man_free,
        "dump_path": dump_path,
        "hedge_w_mean": hedge.w.mean(axis=0).tolist(),
        "hedge_w_deep": hedge.w[deep_mask].mean(axis=0).tolist(),
        "variants": list(("combined", "probe_frozen", "rls_online",
                          "trans_ema")),
    }


def _stagec_worker(args) -> dict:
    (shards, routing, seed, frac, held_group, evict_list, sigma_path,
     workdir, deep_lo, deep_hi, wid) = args
    c = stage0.Corpus(shards, routing, seed, frac)
    sigma = np.asarray(json.load(open(sigma_path))["sigma"], np.float32)
    from elb_train import glm_router
    bank = glm_router.RouterBank.from_npz(
        Path(workdir) / "router_bank.npz")
    r = _stagec_eval_seqs(Path(workdir), c, sigma, bank.bias,
                          held_group, set(evict_list), deep_lo, deep_hi,
                          c.train_keys)
    if r["dump_path"]:
        newp = str(Path(workdir) / f"preds_stagec_combined_p{wid}.npz")
        Path(r["dump_path"]).rename(newp)
        r["dump_path"] = newp
    return r


def run_stagec(c: stage0.Corpus, sigma, bias, work: Path,
               evict_max_seqs: int, deep_lo: int, deep_hi: int,
               jobs: int, sigma_path: str, shards: str, routing: str,
               seed: int, frac: float) -> dict:
    held = sorted(int(s) for s in c.held_keys)
    evict_keys = set(held[:evict_max_seqs])
    if jobs <= 1:
        partials = [_stagec_eval_seqs(work, c, sigma, bias, held,
                                      evict_keys, deep_lo, deep_hi,
                                      c.train_keys)]
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        groups = _partition(held, jobs)
        tasks = [(shards, routing, seed, frac, g,
                  sorted(evict_keys & set(g)), sigma_path, str(work),
                  deep_lo, deep_hi, i) for i, g in enumerate(groups)]
        with cf.ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=mp.get_context("spawn")) as ex:
            partials = list(ex.map(_stagec_worker, tasks))
    out = _finalize_partials(c, work, partials)
    out["jobs"] = max(1, jobs)
    return out


def _finalize_partials(c: stage0.Corpus, work: Path,
                       partials: list[dict]) -> dict:
    variants = partials[0]["variants"]
    pool_true = sum(p["pool_true"] for p in partials)
    addr_n = sum(p["addr_n"] for p in partials)
    out: dict = {"pool": {}, "manifest": {}, "hedge": {}}
    for v in variants:
        out["pool"][v] = {}
        for k in POOL_KS:
            hit = sum(p["pool_hit"][f"{v}|{k}"] for p in partials)
            ah = sum(p["addr_hit"][f"{v}|{k}"] for p in partials)
            out["pool"][v][f"k{k}"] = {
                "coverage": hit / max(1, pool_true),
                "pool_size": c.K + k,
                "addressable_recovery": ah / max(1, addr_n)}
    mf = {alt: [sum(p["man_free"][alt][i] for p in partials)
                for i in range(3)]
          for alt in partials[0]["man_free"]}
    for alt, (cov, size, n) in mf.items():
        out["manifest"][alt] = {"cov_deep": cov / max(1, n),
                                "avg_size": size / max(1, n)}
    base = out["manifest"]["prev_union"]["avg_size"]
    for d in MANIFEST_DS:
        cov = sum(p["man"][str(d)][0] for p in partials)
        n = sum(p["man"][str(d)][1] for p in partials)
        out["manifest"][f"combined_topup{d}"] = {
            "cov_deep": cov / max(1, n), "avg_size": base + d}
    # hedge weights: sequence-count-weighted mean over workers
    wsum = np.zeros(len(BANK))
    wdeep = np.zeros(len(BANK))
    for p in partials:
        wsum += np.asarray(p["hedge_w_mean"])
        wdeep += np.asarray(p["hedge_w_deep"])
    out["hedge"]["members"] = list(BANK)
    out["hedge"]["w_mean"] = (wsum / len(partials)).tolist()
    out["hedge"]["w_deep"] = (wdeep / len(partials)).tolist()
    # merge evict dumps
    parts = [p["dump_path"] for p in partials if p["dump_path"]]
    if parts:
        merged = {k: np.concatenate([np.load(pp)[k] for pp in parts])
                  for k in ("seq", "pos", "layer", "expert", "prob",
                            "label", "occ_prev", "nextdist")}
        p = work / "preds_stagec_combined.npz"
        np.savez_compressed(p, **merged)
        for pp in parts:
            if pp != str(p):
                Path(pp).unlink()
        out["evict_dump"] = str(p)
    out["addressable_slots"] = addr_n
    return out


# ── verdict ──────────────────────────────────────────────────────────────────

def evaluate_stagec_verdict(ev: dict, evict: dict | None,
                            stageb_results: dict | None) -> dict:
    """The mixture is judged against (a) the best free member and (b) the
    frozen fitted probe, on the same pass; its deployed numbers become
    the bar for the admitted capacity candidate."""
    pool = ev["pool"]

    def cov(v, k):
        return pool[v][f"k{k}"]["coverage"]

    out = {
        "pool32_combined": cov("combined", 24),
        "pool32_probe_frozen": cov("probe_frozen", 24),
        "pool32_rls_online": cov("rls_online", 24),
        "pool32_trans_ema": cov("trans_ema", 24),
        "combined_beats_free": cov("combined", 24) > cov("trans_ema", 24),
        "combined_vs_probe": cov("combined", 24) - cov("probe_frozen", 24),
        "rls_matches_probe": abs(cov("rls_online", 24)
                                 - cov("probe_frozen", 24)) < 0.01,
        "manifest_topup32": ev["manifest"]["combined_topup32"]["cov_deep"],
        "hedge_w_deep": dict(zip(ev["hedge"]["members"],
                                 [round(x, 4) for x in
                                  ev["hedge"]["w_deep"]])),
    }
    if evict:
        rows = evict["rows"].get("stagec_combined", {})
        for cap, r in rows.items():
            out[f"evict_capture_{cap}"] = r["capture"]
    out["capacity_challenger_bar"] = {
        "pool32_coverage": out["pool32_combined"],
        "manifest_topup32": out["manifest_topup32"],
        "note": "the admitted §3-P11(4)(iii) capacity candidate must "
                "beat THESE at matched compute (<=50 us/round), else "
                "killed",
    }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--split-seed", type=int, default=35)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--do", default="run,evict,verdict")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--device", default="",
                    help="e.g. cuda:0 — run the dense torch engine on "
                         "one GPU (jobs ignored); empty = CPU path")
    ap.add_argument("--evict-scorer", default="rls_online",
                    choices=("rls_online", "combined", "probe_frozen"),
                    help="protection scorer for the evict dump; "
                         "rls_online = the DEPLOYABLE form (one "
                         "12-feature dot per candidate, weights updated "
                         "free online — b-speed, c-freshness)")
    ap.add_argument("--sigma", default="build/recur/p11-stage0/sigma.json")
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--evict-max-seqs", type=int, default=24)
    ap.add_argument("--evict-caps", default="700,1052")
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--stageb-workdir", default="build/recur/p11-stageb",
                    help="source of the frozen probe + router bank")
    ap.add_argument("--workdir", default="build/recur/p11-stagec")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.do.split(",") if s.strip()]
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    # the frozen probe + router bank come from the stageb workdir
    for fn in ("probe_full.npy", "router_bank.npz"):
        src = Path(a.stageb_workdir) / fn
        dst = work / fn
        if not dst.is_file():
            dst.write_bytes(src.read_bytes())

    c = stage0.Corpus(a.shards, a.routing, a.split_seed, a.held_fraction)
    if a.limit_seqs:
        c.train_keys = c.train_keys[: max(2, a.limit_seqs * 3)]
        c.held_keys = c.held_keys[: a.limit_seqs]
    sigma = np.asarray(json.load(open(a.sigma))["sigma"], np.float32)
    from elb_train import glm_router
    bank = glm_router.RouterBank.from_npz(work / "router_bank.npz")

    rpath = work / "results.json"
    results: dict = json.load(open(rpath)) if rpath.is_file() else {}
    results["config"] = vars(a)

    def _checkpoint():
        json.dump(results, open(rpath, "w"), indent=1)

    if "run" in steps:
        t0 = time.time()
        if a.device:
            from elb_train import stagec_gpu
            held = sorted(int(s) for s in c.held_keys)
            evict_keys = set(held[: a.evict_max_seqs])
            partial = stagec_gpu.gpu_eval_seqs(
                work, c, sigma, bank.bias, held, evict_keys,
                a.deep_lo, a.deep_hi, c.train_keys, device=a.device,
                evict_scorer=a.evict_scorer)
            if "ridge_w" in partial:
                np.savez(work / "ridge_evictor.npz",
                         w=np.asarray(partial.pop("ridge_w")),
                         features=list(FEATURES) + ["bias"],
                         decay=RLS_DECAY, refresh=RLS_REFRESH)
            results["eval"] = _finalize_partials(c, work, [partial])
            results["eval"]["evict_scorer"] = a.evict_scorer
            results["eval"]["ridge_evictor"] = str(
                work / "ridge_evictor.npz")
        else:
            results["eval"] = run_stagec(
                c, sigma, bank.bias, work, a.evict_max_seqs, a.deep_lo,
                a.deep_hi, a.jobs, a.sigma, a.shards, a.routing,
                a.split_seed, a.held_fraction)
        results["eval"]["seconds"] = round(time.time() - t0)
        _checkpoint()
    if "evict" in steps:
        dump = (results.get("eval") or {}).get("evict_dump") or \
            str(work / "preds_stagec_combined.npz")
        if Path(dump).is_file():
            caps = [int(x) for x in a.evict_caps.split(",")]
            results["evict"] = stage0.run_evict(
                {"stagec_combined": dump}, caps)
            _checkpoint()
    if "verdict" in steps:
        results["verdict"] = evaluate_stagec_verdict(
            results["eval"], results.get("evict"), None)
        print(json.dumps(results["verdict"], indent=1), flush=True)
        _checkpoint()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
    print(f"[stagec] wrote {rpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
