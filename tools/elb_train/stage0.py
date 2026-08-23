"""P11 Stage-0 driver: b0_prev gap decomposition + free-signal-bank bench
(spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(6); PLAN.md Phase 30 P11.a).

Offline, zero-training, CPU-only. Two streaming passes over the EPM-5
corpus:

  pass 1 (TRAIN split): warm the stateful bank members (token memo,
         transition EMA) + estimate per-layer score-drift sigma.
  pass 2 (HELD split):  per committed position — b0_prev miss-mass
         decomposition (within-pool / novel / boundary), bank member
         predict-then-update skill eval, per-verify-chunk manifest
         coverage, and the evict_sim demand dump.

`--do verify` runs the corpus-integrity checks alone and MUST be green
before any result is trusted:
  (a) routing-stream top-8 == shard labels at every labeled held cell
      (retires the run_XXX.bin <-> run_idx join caveat),
  (b) b0_prev recall@8 recomputed with routing-sourced tops matches the
      committed bar (studies/epm/epm5-corpus/bars_b0prev.json, 0.30426),
  (c) glm_router.rank_experts on the stored FP16 logits reproduces the
      stored top-8 sets (>= 99%; the engine selected in F32 and the dump
      stores FP16 logits — measured: ~0.97% of rows flip a boundary
      expert, every flip within 1.5e-4 sel units = exactly fp16 noise,
      so the score space is faithful and boundary flips are noise).

Usage: see spec plan; canonical invocation in studies/epm/p11-stage0/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import (dataset, glm_router, recurrence,  # noqa: E402
                       routing_reader, signal_bank, stage0_decompose)

DEFAULT_ROUTER_CKPT = "/srv/models/lukealonso/GLM-5.2-NVFP4"
EVICT_TOPM = 56          # protect budget ~= prev-chunk-union avg (55.4)
EVICT_DUMP_MEMBERS = ("b0_prev", "trail16_freq", "score_ema8",
                      "token_memo_warm", "trans_ema_warm")


# ── Corpus plumbing ──────────────────────────────────────────────────────────

class Corpus:
    """Light index over shards + routing runs: block coordinates, committed
    timelines, per-run memmap views, split membership."""

    def __init__(self, shard_dir: str | Path, routing_dir: str | Path,
                 split_seed: int, held_fraction: float):
        self.shard_dir = Path(shard_dir)
        self.routing_dir = Path(routing_dir)
        self.index = dataset.load_index(self.shard_dir)
        self.moe_layers = np.asarray(self.index["moe_layers"], np.int32)
        self.J = len(self.moe_layers)
        self.K = int(self.index["topk"])
        self.G = int(self.index["max_gamma"])
        self.E = int(self.index["n_experts"])
        allk = sorted({k for sh in self.index["shards"]
                       for k in sh["seq_keys"]})
        self.train_keys, self.held_keys = dataset.sequence_split(
            allk, held_fraction, seed=split_seed)
        # run_idx -> routing file (index run_units: [orig_path, run_idx];
        # the corpus relocation flattened build/epm5-corpus/run_XXX ->
        # routing/run_XXX.bin, verified by check (a)).
        self.run_files: dict[int, Path] = {}
        for orig, ridx in self.index["run_units"]:
            name = Path(orig).name + ".bin"
            p = self.routing_dir / name
            if p.is_file():
                self.run_files[int(ridx)] = p
        self._views: dict[int, routing_reader.RunView] = {}
        self._seq_maps: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def view(self, run_idx: int) -> routing_reader.RunView:
        v = self._views.get(run_idx)
        if v is None:
            v = routing_reader.open_run(self.run_files[run_idx])
            if not np.array_equal(v.layers, self.moe_layers):
                raise ValueError(f"{v.path}: layer set differs from the "
                                 f"shard index moe_layers")
            self._views[run_idx] = v
        return v

    def seq_map(self, run_idx: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """seq_id -> (sorted pos array, record indices) for one run."""
        m = self._seq_maps.get(run_idx)
        if m is None:
            v = self.view(run_idx)
            m = {}
            for seq_id, idx in routing_reader.iter_sequences(v):
                m[seq_id] = (v.pos[idx].astype(np.int64), idx)
            self._seq_maps[run_idx] = m
        return m

    def drop_run(self, run_idx: int) -> None:
        """Release a run's memmap + maps (streaming passes keep one open)."""
        self._views.pop(run_idx, None)
        self._seq_maps.pop(run_idx, None)

    def lookup_records(self, run_idx: int, seq_id: int,
                       positions: np.ndarray) -> np.ndarray:
        """Record indices for absolute decode positions of one sequence;
        raises if any position has no routing record."""
        pos_arr, idx = self.seq_map(run_idx)[seq_id]
        loc = np.searchsorted(pos_arr, positions)
        bad = (loc >= len(pos_arr)) | (pos_arr[np.minimum(
            loc, len(pos_arr) - 1)] != positions)
        if np.any(bad):
            p = positions[np.nonzero(bad)[0][0]]
            raise KeyError(f"run {run_idx} seq {seq_id}: no routing record "
                           f"at position {p}")
        return idx[loc]

    def iter_shards(self, seq_keys) -> "list[tuple[int, dict]]":
        keys = {int(s) for s in seq_keys}
        out = []
        for si, sh in enumerate(self.index["shards"]):
            if keys.intersection(sh["seq_keys"]):
                out.append((si, sh))
        return out

    def load_shard(self, si: int, keys: tuple[str, ...]) -> dict:
        z = np.load(self.shard_dir / self.index["shards"][si]["path"])
        return {k: z[k] for k in keys}

    def committed_timeline(self, seq_keys) -> dict[int, list]:
        """seq_key -> list of (abs_pos, si, row, k) committed+labeled cells
        sorted by position, plus per-cell block id for chunk grouping.
        Committed = k <= accepted_len AND label_mask[k] (recurrence.py
        contract; blocks tile contiguously)."""
        keys = {int(s) for s in seq_keys}
        seqs: dict[int, list] = {}
        for si, sh in self.iter_shards(keys):
            z = self.load_shard(si, ("seq_key", "anchor_pos", "accepted_len",
                                     "label_mask"))
            sk = z["seq_key"].astype(np.int64)
            sel = np.nonzero(np.isin(sk, list(keys)))[0]
            for r in sel:
                acc = int(z["accepted_len"][r])
                base = int(z["anchor_pos"][r])
                lm = z["label_mask"][r]
                for k in range(min(acc + 1, self.G)):
                    if lm[k]:
                        seqs.setdefault(int(sk[r]), []).append(
                            (base + k, si, int(r), k))
        for s in seqs:
            seqs[s].sort()
        return seqs


# ── cell gathering ───────────────────────────────────────────────────────────

def gather_split_cells(c: Corpus, seq_keys, recur_keys: set[int] = frozenset()
                       ) -> dict[int, dict]:
    """Per-sequence cell streams for one split.

    seq_key -> {run_idx, seq_id,
      pos/token/chunk [T]      — the COMMITTED timeline (k <= accepted_len
                                 AND label_mask[k], recurrence.py contract):
                                 the real decode trajectory the online
                                 members live on,
      blocks: list per verify chunk (block, anchor-sorted) of
        {"full_pos", "full_token"}  — ALL labeled positions incl. rejected
                                 speculative rows: the verify forward
                                 touches their experts too, so the
                                 deployed manifest target U_cur is the
                                 union over THESE (matches
                                 compute_deployed_bar / the 0.888 bar),
      recur (recur_keys only): label/occ_prev/nextdist [T, J, K] from the
                                 recur_%05d.npz sidecars (evict_sim feed)}.

    Input token at cell k: the anchor token for k == 0, else draft token
    k-1 (for committed cells the accepted prefix IS the token stream; for
    speculative rows the draft tokens are what the verify actually fed)."""
    keys = {int(s) for s in seq_keys}
    recur_keys = {int(s) for s in recur_keys}
    seqs: dict[int, dict] = {}
    for si, sh in c.iter_shards(keys):
        z = c.load_shard(si, ("seq_key", "seq_id", "run_idx", "anchor_pos",
                              "accepted_len", "label_mask", "anchor_token",
                              "draft_tokens", "gamma", "confidences"))
        sk = z["seq_key"].astype(np.int64)
        rows = np.nonzero(np.isin(sk, list(keys)))[0]
        rc = None
        if recur_keys and {int(s) for s in sk[rows]} & recur_keys:
            rc = recurrence.load_recur_sidecar(c.shard_dir, si)
            if rc is None:
                raise FileNotFoundError(
                    f"recur_{si:05d}.npz missing (build_recurrence_sidecars)")
        for r in rows:
            key = int(sk[r])
            s = seqs.setdefault(key, {
                "run_idx": int(z["run_idx"][r]),
                "seq_id": int(z["seq_id"][r]),
                "pos": [], "token": [], "conf": [], "anchor": [],
                "blocks": {},
                "recur": [] if key in recur_keys else None,
            })
            acc = int(z["accepted_len"][r])
            base = int(z["anchor_pos"][r])
            gm = int(z["gamma"][r])
            lm = z["label_mask"][r]

            def tok_at(k: int) -> int:
                return (int(z["anchor_token"][r]) if k == 0
                        else int(z["draft_tokens"][r, k - 1]))

            full_pos, full_tok, full_conf = [], [], []
            for k in range(min(gm, c.G)):
                if not lm[k]:
                    continue
                full_pos.append(base + k)
                full_tok.append(tok_at(k))
                full_conf.append(float(z["confidences"][r, k]))
                if k <= acc:
                    s["pos"].append(base + k)
                    s["token"].append(tok_at(k))
                    s["conf"].append(float(z["confidences"][r, k]))
                    s["anchor"].append(base)
                    if s["recur"] is not None:
                        s["recur"].append((rc["recur16"][r, k],
                                           rc["occ_prev"][r, k],
                                           rc["nextdist"][r, k]))
            if full_pos:
                s["blocks"][base] = {
                    "anchor": base,
                    "full_pos": np.asarray(full_pos, np.int64),
                    "full_token": np.asarray(full_tok, np.int32),
                    "full_conf": np.asarray(full_conf, np.float32)}
    for key, s in seqs.items():
        order = np.argsort(np.asarray(s["pos"], np.int64), kind="stable")
        s["pos"] = np.asarray(s["pos"], np.int64)[order]
        s["token"] = np.asarray(s["token"], np.int32)[order]
        s["conf"] = np.nan_to_num(
            np.asarray(s["conf"], np.float32)[order])
        anchors = np.asarray(s["anchor"], np.int64)[order]
        uniq = np.array(sorted(s["blocks"]), np.int64)
        s["chunk"] = np.searchsorted(uniq, anchors).astype(np.int64)
        s["blocks"] = [s["blocks"][a] for a in uniq]
        del s["anchor"]
        if s["recur"] is not None:
            lab = np.stack([t[0] for t in s["recur"]])[order]
            occ = np.stack([t[1] for t in s["recur"]])[order]
            nxt = np.stack([t[2] for t in s["recur"]])[order]
            s["recur"] = {"label": lab, "occ_prev": occ, "nextdist": nxt}
    return seqs


def _by_run(seqs: dict[int, dict]) -> list[tuple[int, list[int]]]:
    runs: dict[int, list[int]] = {}
    for key, s in seqs.items():
        runs.setdefault(s["run_idx"], []).append(key)
    return sorted((r, sorted(ks)) for r, ks in runs.items())


def _sel_scores(logits_f16: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """FP16 logits [T, J, E] -> noaux_tc selection scores f32 (the score
    space verified to reproduce the stored top-8, verify check (c))."""
    return glm_router.stable_sigmoid(
        logits_f16.astype(np.float32)) + bias[None]


def topm_membership(pred: np.ndarray, m: int, sparse: bool) -> np.ndarray:
    """bool [J, E]: expert in the top-m of pred, per layer. Sparse
    (count-type) scores predict NOTHING at 0 — argpartition must not
    promote arbitrary zero-score experts into the pool; dense scores
    (sel-space, possibly negative) rank every expert but an all-zero row
    (unwarmed state) still predicts nothing."""
    J, E = pred.shape
    topm = np.argpartition(pred, E - m, axis=1)[:, E - m:]
    memb = np.zeros((J, E), bool)
    np.put_along_axis(memb, topm, True, axis=1)
    if sparse:
        memb &= pred > 0
    else:
        memb &= pred.any(axis=1)[:, None]
    return memb


# ── warm pass (train split) ──────────────────────────────────────────────────

def warm_pass(c: Corpus, warm_members, sigma_est, sigma_seqs: int,
              bias: np.ndarray) -> None:
    seqs = gather_split_cells(c, c.train_keys)
    sigma_left = sigma_seqs
    n_done = 0
    t0 = time.time()
    for run_idx, keys in _by_run(seqs):
        v = c.view(run_idx)
        for key in keys:
            s = seqs[key]
            recs = c.lookup_records(run_idx, s["seq_id"], s["pos"])
            tops = np.ascontiguousarray(v.top_ids[recs])      # [T, J, K]
            for m in warm_members:
                m.reset_seq()
            for t in range(len(tops)):
                for m in warm_members:
                    m.update(tops[t], None, None, int(s["token"][t]))
            if sigma_left > 0:
                sel = _sel_scores(v.logits(recs), bias)
                for j in range(c.J):
                    sigma_est.add_sequence_layer(j, sel[:, j])
                sigma_left -= 1
            n_done += 1
            if n_done % 50 == 0:
                print(f"[stage0] warm {n_done}/{len(seqs)} seqs "
                      f"({time.time()-t0:.0f}s)", flush=True)
        c.drop_run(run_idx)


# ── eval pass (held split) ───────────────────────────────────────────────────

class BankEval:
    """Model-side skill accumulators for one member."""

    def __init__(self, name: str, n_layers: int, pool_ms: list[int],
                 sparse: bool):
        self.name = name
        self.pool_ms = pool_ms
        self.sparse = sparse
        self.hit8 = np.zeros(n_layers, np.int64)
        self.cov = {m: np.zeros(n_layers, np.int64) for m in pool_ms}
        self.n_true = np.zeros(n_layers, np.int64)

    def add(self, pred: np.ndarray, tops: np.ndarray) -> None:
        """pred [J, E] scores, tops [J, K] true ids."""
        K = tops.shape[1]
        t64 = tops.astype(np.int64)
        self.n_true += K
        for m in self.pool_ms:
            memb = topm_membership(pred, m, self.sparse)
            hits = np.take_along_axis(memb, t64, axis=1).sum(axis=1)
            self.cov[m] += hits
            if m == K:
                self.hit8 += hits

    def result(self, deep_idx: np.ndarray) -> dict:
        def frac(num, idx):
            return float(num[idx].sum() / max(1, self.n_true[idx].sum()))
        out = {"recall8_all": frac(self.hit8, slice(None)),
               "recall8_deep": frac(self.hit8, deep_idx)}
        for m in self.pool_ms:
            out[f"cov{m}_all"] = frac(self.cov[m], slice(None))
            out[f"cov{m}_deep"] = frac(self.cov[m], deep_idx)
        return out


class ManifestEval:
    """Deployed manifest proxy for one member: per verify chunk (with a
    predecessor), union of top-M state predictions at deep layers vs the
    chunk's true union U_cur."""

    def __init__(self, name: str, pool_ms: list[int]):
        self.name = name
        self.pool_ms = pool_ms
        self.cov_sum = {m: 0.0 for m in pool_ms}
        self.size_sum = {m: 0.0 for m in pool_ms}
        self.n = 0

    def add(self, manifest: dict[int, np.ndarray], ucur: np.ndarray) -> None:
        """manifest: pool_m -> bool [Jd, E]; ucur: bool [Jd, E]."""
        u_n = ucur.sum(axis=1)
        ok = u_n > 0
        if not np.any(ok):
            return
        self.n += int(ok.sum())
        for m in self.pool_ms:
            mm = manifest[m]
            inter = (mm & ucur).sum(axis=1)
            self.cov_sum[m] += float((inter[ok] / u_n[ok]).sum())
            self.size_sum[m] += float(mm[ok].sum(axis=1).sum())

    def result(self) -> dict:
        return {f"m{m}": {
            "cov_deep": self.cov_sum[m] / max(1, self.n),
            "avg_size": self.size_sum[m] / max(1, self.n),
        } for m in self.pool_ms}


def eval_pass(c: Corpus, bank, sigma: np.ndarray, bias: np.ndarray,
              cfg) -> dict:
    """Held-split pass: decomposition + bank skill + manifest proxy +
    evict_sim demand dumps. Returns the results fragment."""
    held = sorted(int(s) for s in c.held_keys)
    evict_keys = set(held[: cfg.evict_max_seqs])
    seqs = gather_split_cells(c, held, recur_keys=evict_keys)
    deep_idx = np.array([i for i, l in enumerate(c.moe_layers)
                         if cfg.deep_lo <= l <= cfg.deep_hi])
    decomp = stage0_decompose.DecompAccum(
        windows=cfg.windows, cs=cfg.cs, n_layers=c.J)
    evals = {m.name: BankEval(m.name, c.J, cfg.pool_ms, m.sparse)
             for m in bank}
    for d in cfg.xlayer_deltas:
        evals[f"xlayer_d{d}"] = BankEval(f"xlayer_d{d}", c.J, cfg.pool_ms,
                                         sparse=False)
    manifests = {m.name: ManifestEval(m.name, cfg.pool_ms) for m in bank}
    man_base = ManifestEval("prev_chunk_union", cfg.pool_ms)
    dump: dict[str, dict[str, list]] = {
        n: {k: [] for k in ("seq", "pos", "layer", "expert", "prob",
                            "label", "occ_prev", "nextdist")}
        for n in EVICT_DUMP_MEMBERS}
    lay_grid = np.repeat(np.arange(c.J, dtype=np.int16), c.K)

    n_done = 0
    t0 = time.time()
    for run_idx, keys in _by_run(seqs):
        v = c.view(run_idx)
        for key in keys:
            s = seqs[key]
            T = len(s["pos"])
            recs = c.lookup_records(run_idx, s["seq_id"], s["pos"])
            tops = np.ascontiguousarray(v.top_ids[recs])       # [T, J, K]
            top_w = np.ascontiguousarray(v.top_w[recs])        # [T, J, K]
            sel = _sel_scores(v.logits(recs), bias)            # [T, J, E]
            chunk = s["chunk"]

            # 1. decomposition (t >= 1, per layer)
            for j in range(c.J):
                decomp.add_sequence_layer(
                    j, tops[:, j], top_w[:, j], sel[:, j],
                    float(sigma[j]), chunk)

            # 2. cross-layer coherence (same position, layer j-delta -> j)
            for d in cfg.xlayer_deltas:
                ev = evals[f"xlayer_d{d}"]
                for t in range(T):
                    pred = np.zeros((c.J, c.E), np.float32)
                    pred[d:] = sel[t, :-d]
                    ev.add(pred, tops[t])

            # 3. online bank + manifest + evict dump, per verify chunk
            for m in bank:
                m.reset_seq()
            is_evict = s["recur"] is not None
            prev_ucur: np.ndarray | None = None
            for ci, blk in enumerate(s["blocks"]):
                # manifest predictions from PRE-chunk state; the chunk's
                # draft tokens are known at manifest time.
                man_at: dict[str, dict] = {}
                if ci > 0:
                    for m in bank:
                        if isinstance(m, signal_bank.TokenMemo):
                            toks = sorted({int(t)
                                           for t in blk["full_token"]})
                            scores = [m.predict(t) for t in toks]
                        else:
                            scores = [m.predict(int(blk["full_token"][0]))]
                        man = {}
                        for pm in cfg.pool_ms:
                            u = np.zeros((len(deep_idx), c.E), bool)
                            for sc in scores:
                                u |= topm_membership(sc[deep_idx], pm,
                                                     m.sparse)
                            man[pm] = u
                        man_at[m.name] = man
                # U_cur over ALL labeled rows of the chunk (incl. rejected
                # speculative rows — the verify forward touches them).
                frecs = c.lookup_records(run_idx, s["seq_id"],
                                         blk["full_pos"])
                ftops = v.top_ids[frecs][:, deep_idx].astype(np.int64)
                ucur = np.zeros((len(deep_idx), c.E), bool)
                for ft in ftops:
                    np.put_along_axis(ucur, ft, True, axis=1)
                if ci > 0:
                    for m in bank:
                        manifests[m.name].add(man_at[m.name], ucur)
                    if prev_ucur is not None:
                        man_base.add({pm: prev_ucur for pm in cfg.pool_ms},
                                     ucur)
                prev_ucur = ucur
                # per-position online eval over the chunk's COMMITTED cells
                for t in np.nonzero(chunk == ci)[0]:
                    tok = int(s["token"][t])
                    for m in bank:
                        pred = m.predict(tok)
                        evals[m.name].add(pred, tops[t])
                        if is_evict and m.name in dump:
                            dm = dump[m.name]
                            prot = topm_membership(pred, EVICT_TOPM,
                                                   m.sparse)
                            t64 = tops[t].astype(np.int64)
                            pb = np.take_along_axis(
                                prot, t64, axis=1).astype(np.float32)
                            dm["seq"].append(np.full(c.J * c.K, key,
                                                     np.int64))
                            dm["pos"].append(np.full(c.J * c.K,
                                                     s["pos"][t], np.int32))
                            dm["layer"].append(lay_grid.copy())
                            dm["expert"].append(
                                tops[t].astype(np.int16).ravel())
                            dm["prob"].append(pb.ravel())
                            dm["label"].append(
                                s["recur"]["label"][t].ravel())
                            dm["occ_prev"].append(
                                s["recur"]["occ_prev"][t].ravel())
                            dm["nextdist"].append(
                                s["recur"]["nextdist"][t].ravel())
                    for m in bank:
                        m.update(tops[t], top_w[t], sel[t], tok)
            n_done += 1
            if n_done % 10 == 0:
                print(f"[stage0] eval {n_done}/{len(seqs)} seqs "
                      f"({time.time()-t0:.0f}s)", flush=True)
        c.drop_run(run_idx)

    # finalize
    out: dict = {
        "decompose": decomp.summarize(c.moe_layers, cfg.deep_lo,
                                      cfg.deep_hi, sigma),
        "bank": {}, "manifest": {},
    }
    b0 = evals["b0_prev"].result(deep_idx)
    for name, ev in evals.items():
        r = ev.result(deep_idx)
        for metric in ("recall8_all", "recall8_deep"):
            base = b0[metric]
            r["S_" + metric] = ((r[metric] - base) / (1.0 - base)
                                if base < 1.0 else 0.0)
        member = next((m for m in bank if m.name == name), None)
        if member is not None:
            r["cost_us"] = member.cost_us
            r["state_bytes"] = member.state_bytes
        out["bank"][name] = r
    base_man = man_base.result()
    out["manifest"]["prev_chunk_union"] = base_man
    bm = base_man[f"m{cfg.pool_ms[0]}"]
    for name, me in manifests.items():
        r = me.result()
        for pm in cfg.pool_ms:
            cov = r[f"m{pm}"]["cov_deep"]
            bcov = bm["cov_deep"]
            r[f"m{pm}"]["S_cov"] = ((cov - bcov) / (1.0 - bcov)
                                    if bcov < 1.0 else 0.0)
        out["manifest"][name] = r
    out["evict_dumps"] = {}
    for name, dm in dump.items():
        if not dm["seq"]:
            continue
        arrs = {k: np.concatenate(vs) for k, vs in dm.items()}
        path = Path(cfg.workdir) / f"preds_{name}.npz"
        np.savez_compressed(path, **arrs)
        out["evict_dumps"][name] = str(path)
    return out


# ── verify ───────────────────────────────────────────────────────────────────

def run_verify(c: Corpus, router_bank, bars_path: str | None,
               rank_sample: int = 200_000) -> dict:
    """Checks (a)/(b)/(c) of the module docstring over the HELD split."""
    rng = np.random.default_rng(0)
    t0 = time.time()
    held = set(c.held_keys)

    n_cells = 0
    n_mismatch = 0
    # (b) accumulators: recall@8 of prev-sidecar tops vs ROUTING tops.
    inter_sum = 0.0
    cell_n = 0
    # (c) accumulators
    rank_checked = 0
    rank_set_ok = 0
    rank_ord_ok = 0

    for si, sh in c.iter_shards(held):
        z = c.load_shard(si, ("seq_key", "seq_id", "run_idx", "anchor_pos",
                              "label_mask", "labels_top_ids"))
        prev_path = c.shard_dir / f"prev_top_ids_{si:05d}.npy"
        prev = np.load(prev_path)
        sk = z["seq_key"].astype(np.int64)
        rows = np.nonzero(np.isin(sk, list(held)))[0]
        for r in rows:
            run_idx = int(z["run_idx"][r])
            seq_id = int(z["seq_id"][r])
            base = int(z["anchor_pos"][r])
            lab = np.nonzero(z["label_mask"][r])[0]
            if lab.size == 0:
                continue
            recs = c.lookup_records(run_idx, seq_id, base + lab)
            v = c.view(run_idx)
            rtops = v.top_ids[recs]                     # [n, J, K]
            stops = z["labels_top_ids"][r, lab]         # [n, J, K]
            n_cells += rtops.shape[0] * c.J
            n_mismatch += int(np.any(rtops != stops, axis=-1).sum())
            # (b): prev sidecar vs routing tops
            P = prev[r, lab]                            # [n, J, K]
            m = (P[..., :, None] == rtops[..., None, :]) & (
                rtops[..., None, :] >= 0)
            inter = m.any(axis=-2).sum(axis=-1)
            ntrue = (rtops >= 0).sum(axis=-1)
            ok = ntrue > 0
            inter_sum += float((inter[ok] / ntrue[ok]).sum())
            cell_n += int(ok.sum())
            # (c): rank reproduction on a subsample of (cell) rows
            if rank_checked < rank_sample:
                take = min(4, rtops.shape[0])
                pick = rng.choice(rtops.shape[0], take, replace=False)
                lg = v.logits(recs[pick])               # [take, J, E] f16
                pred = glm_router.rank_experts(lg, router_bank.bias, c.K)
                for a_row, b_row in zip(pred.reshape(-1, c.K),
                                        rtops[pick].reshape(-1, c.K)):
                    rank_checked += 1
                    if np.array_equal(a_row, b_row):
                        rank_ord_ok += 1
                        rank_set_ok += 1
                    elif set(a_row.tolist()) == set(b_row.tolist()):
                        rank_set_ok += 1

    recall8 = inter_sum / max(1, cell_n)
    out = {
        "held_seqs": len(c.held_keys),
        "labeled_layer_cells": n_cells,
        "join_mismatch_cells": n_mismatch,
        "b0prev_recall8_routing": recall8,
        "rank_rows_checked": rank_checked,
        "rank_set_agreement": rank_set_ok / max(1, rank_checked),
        "rank_ordered_agreement": rank_ord_ok / max(1, rank_checked),
        "seconds": round(time.time() - t0),
    }
    if bars_path:
        bar = json.load(open(bars_path))["b0prev_recall8"]
        out["b0prev_recall8_bar"] = bar
        out["bar_delta"] = recall8 - bar
    ok_a = n_mismatch == 0
    ok_b = bars_path is None or abs(out["bar_delta"]) < 1e-3
    ok_c = out["rank_set_agreement"] >= 0.99
    out["verify_ok"] = bool(ok_a and ok_b and ok_c)
    return out


# ── evict proxy ──────────────────────────────────────────────────────────────

def _evict_worker(args: tuple) -> tuple:
    """One (dump, cap, policy) simulation in a worker process. The sim is
    a strictly sequential cache-state chain (each decision depends on all
    prior evictions) — no intra-trace parallelism, GPU included; the win
    is that all (member x cap x policy) sims are independent."""
    path, cap, policy = args
    from elb_train import evict_sim
    trace = evict_sim.build_trace(dict(np.load(path)))
    return path, cap, policy, evict_sim.simulate(trace, cap, policy)


def run_evict(dump_paths: dict[str, str], caps: list[int],
              jobs: int = 0) -> dict:
    """evict_sim over the member demand dumps, process-parallel across
    sims (wall = slowest single sim). Member-independent policies
    (lru/prevunion/bridge16/belady) run ONCE on the first dump — every
    dump shares seq/pos/layer/expert/occ_prev/nextdist; only `prob`
    (member protection) differs."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from elb_train import evict_sim
    names = list(dump_paths)
    first_path = dump_paths[names[0]]
    tasks = [(first_path, cap, pol) for cap in caps
             for pol in ("lru", "prevunion", "bridge16", "belady")]
    tasks += [(dump_paths[n], cap, "pred_bridge")
              for n in names for cap in caps]
    jobs = jobs or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    res: dict[tuple, float] = {}
    # spawn: callers may have run OpenMP (lightgbm) — forking then
    # deadlocks children on the inherited OMP runtime.
    with cf.ProcessPoolExecutor(
            max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        for path, cap, pol, hit in ex.map(_evict_worker, tasks):
            res[(path, cap, pol)] = hit
            print(f"[stage0] evict {Path(path).stem} cap {cap} "
                  f"{pol}: {hit:.4f}", flush=True)

    pred0 = dict(np.load(first_path))
    trace0 = evict_sim.build_trace(pred0)
    out: dict = {
        "caps": caps, "protect_topm": EVICT_TOPM, "rows": {},
        "n_demands": int(len(trace0["key"])),
        "n_seqs": int(len(set(trace0["seq"].tolist()))),
        "baselines": {str(cap): {
            pol: res[(first_path, cap, pol)]
            for pol in ("lru", "prevunion", "bridge16", "belady")}
            for cap in caps},
    }
    for name in names:
        rows = {}
        for cap in caps:
            b = out["baselines"][str(cap)]
            gain = b["belady"] - b["lru"]
            hit = res[(dump_paths[name], cap, "pred_bridge")]
            rows[str(cap)] = {
                "hit": hit,
                "capture": (hit - b["lru"]) / gain if gain > 0 else 0.0,
                "prevunion_capture": ((b["prevunion"] - b["lru"]) / gain
                                      if gain > 0 else 0.0),
            }
        out["rows"][name] = rows
    return out


# ── report ───────────────────────────────────────────────────────────────────

def write_table(results: dict, path: Path) -> None:
    L = []
    cfgd = results.get("config", {})
    L.append("# P11 Stage-0 results\n")
    L.append(f"Config: `{json.dumps(cfgd)}`\n")
    if "verify" in results:
        v = results["verify"]
        L.append(f"Verify: join mismatches {v['join_mismatch_cells']}, "
                 f"b0_prev recall {v['b0prev_recall8_routing']:.5f} "
                 f"(bar delta {v.get('bar_delta', float('nan')):+.2e}), "
                 f"rank-set agreement {v['rank_set_agreement']:.4f}\n")
    d = results.get("decompose")
    if d:
        L.append("\n## Miss-mass decomposition (b0_prev gap)\n")
        for scope in ("all", "deep"):
            a = d[scope]
            L.append(f"\n### {scope} layers — recall@8 "
                     f"{a['recall8']:.4f}, miss mass "
                     f"{a['miss_mass']:.4f}\n")
            L.append("| window x c | f_boundary | f_within | f_novel | "
                     "within headroom (abs recall) |")
            L.append("|---|---|---|---|---|")
            for key, g in a["grid"].items():
                L.append(f"| {key} | {g['f_boundary']:.3f} | "
                         f"{g['f_within']:.3f} | {g['f_novel']:.3f} | "
                         f"{g['within_headroom']:.4f} |")
    b = results.get("bank")
    if b:
        L.append("\n## Free-signal bank — model-side (held split)\n")
        L.append("| member | recall@8 all | S(recall) | recall@8 deep | "
                 "cov@32 all | cov@64 all | cost µs | state B |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, r in b.items():
            L.append(f"| {name} | {r['recall8_all']:.4f} | "
                     f"{r['S_recall8_all']:+.3f} | "
                     f"{r['recall8_deep']:.4f} | "
                     f"{r.get('cov32_all', 0):.4f} | "
                     f"{r.get('cov64_all', 0):.4f} | "
                     f"{r.get('cost_us', '')} | "
                     f"{r.get('state_bytes', '')} |")
    man = results.get("manifest")
    if man:
        L.append("\n## Manifest proxy — deep layers, per verify chunk\n")
        L.append("| member | m | coverage | avg experts/layer | S(cov) |")
        L.append("|---|---|---|---|---|")
        for name, r in man.items():
            for mkey, row in r.items():
                L.append(f"| {name} | {mkey} | {row['cov_deep']:.4f} | "
                         f"{row['avg_size']:.1f} | "
                         f"{row.get('S_cov', 0):+.3f} |")
    ev = results.get("evict")
    if ev:
        L.append("\n## Eviction proxy — Belady-gap capture "
                 f"(protect top-{ev['protect_topm']}, "
                 f"{ev.get('n_seqs', '?')} seqs, "
                 f"{ev.get('n_demands', 0):,} demands)\n")
        L.append("| member | cap | hit | capture | prevunion capture |")
        L.append("|---|---|---|---|---|")
        for name, rows in ev["rows"].items():
            for cap, r in rows.items():
                L.append(f"| {name} | {cap} | {r['hit']:.4f} | "
                         f"{r['capture']*100:.1f}% | "
                         f"{r['prevunion_capture']*100:.1f}% |")
    path.write_text("\n".join(L) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--router-ckpt", default=DEFAULT_ROUTER_CKPT)
    ap.add_argument("--split-seed", type=int, default=35)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--do", default="verify",
                    help="comma list: verify, eval (decompose+bank+"
                         "manifest+dumps in one pass), evict (sims over "
                         "the dumps); decompose/bank/manifest accepted "
                         "as aliases for eval")
    ap.add_argument("--xlayer-deltas", default="1,2,4")
    ap.add_argument("--bars", default="studies/epm/epm5-corpus/"
                    "bars_b0prev.json")
    ap.add_argument("--windows", default="prevpos,prevchunk,trail16,trail64")
    ap.add_argument("--sigma-c", default="0,0.5,1,2",
                    help="boundary thresholds; c=0 disables the boundary "
                         "carve-out (pure window decomposition)")
    ap.add_argument("--pool-m", default="8,16,32,64")
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--sigma-seqs", type=int, default=50)
    ap.add_argument("--limit-seqs", type=int, default=0,
                    help="debug: cap held sequences")
    ap.add_argument("--evict-max-seqs", type=int, default=24)
    ap.add_argument("--evict-caps", default="700,1052")
    ap.add_argument("--workdir", default="build/recur/p11-stage0")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    steps = [s.strip() for s in a.do.split(",") if s.strip()]
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    c = Corpus(a.shards, a.routing, a.split_seed, a.held_fraction)
    print(f"[stage0] corpus: {len(c.index['shards'])} shards, "
          f"{len(c.run_files)} routing runs, J={c.J} E={c.E} K={c.K} "
          f"G={c.G}; split {len(c.train_keys)} train / "
          f"{len(c.held_keys)} held", flush=True)
    missing = [ri for ri, _ in enumerate(c.index["run_units"])
               if ri not in c.run_files]
    if missing:
        raise FileNotFoundError(f"routing files missing for run_idx "
                                f"{missing[:5]}... under {a.routing}")
    if a.limit_seqs:
        c.held_keys = c.held_keys[:a.limit_seqs]

    # Router bank (bias needed for rank reproduction + margins; cached).
    bank_npz = work / "router_bank.npz"
    if bank_npz.is_file():
        bank = glm_router.RouterBank.from_npz(bank_npz)
    else:
        print("[stage0] loading router bank from checkpoint ...", flush=True)
        bank = glm_router.RouterBank.from_checkpoint(
            a.router_ckpt, c.moe_layers)
        bank.save_npz(bank_npz)

    results: dict = {"config": {
        "split_seed": a.split_seed, "held_fraction": a.held_fraction,
        "windows": a.windows, "sigma_c": a.sigma_c, "pool_m": a.pool_m,
        "deep": [a.deep_lo, a.deep_hi], "limit_seqs": a.limit_seqs,
    }}

    if "verify" in steps:
        print("[stage0] verify ...", flush=True)
        vr = run_verify(c, bank, a.bars or None)
        results["verify"] = vr
        print(json.dumps(vr, indent=1), flush=True)
        if not vr["verify_ok"]:
            print("[stage0] VERIFY FAILED — aborting", flush=True)
            if a.out:
                json.dump(results, open(a.out, "w"), indent=1)
            return 1

    want_eval = bool({"eval", "decompose", "bank", "manifest"} & set(steps))
    if want_eval:
        import types
        cfg = types.SimpleNamespace(
            windows=[w.strip() for w in a.windows.split(",") if w.strip()],
            cs=[float(x) for x in a.sigma_c.split(",")],
            pool_ms=[int(x) for x in a.pool_m.split(",")],
            xlayer_deltas=[int(x) for x in a.xlayer_deltas.split(",")],
            deep_lo=a.deep_lo, deep_hi=a.deep_hi,
            evict_max_seqs=a.evict_max_seqs, workdir=str(work))

        # warm pass: train the global tables + estimate sigma.
        sigma_path = work / "sigma.json"
        warm_bank = signal_bank.build_bank(c.J, c.E, warmed=True)
        sig_est = stage0_decompose.SigmaEstimator(c.J)
        print(f"[stage0] warm pass over {len(c.train_keys)} train seqs "
              f"(sigma from {a.sigma_seqs}) ...", flush=True)
        warm_pass(c, warm_bank, sig_est, a.sigma_seqs, bank.bias)
        sigma = sig_est.sigma()
        json.dump({"sigma": sigma.tolist(),
                   "samples": sig_est.count.tolist()},
                  open(sigma_path, "w"))
        print(f"[stage0] sigma: median {np.nanmedian(sigma):.5f} "
              f"range [{np.nanmin(sigma):.5f}, {np.nanmax(sigma):.5f}]",
              flush=True)

        # eval pass: cold members + the warmed tables (kept warm).
        full_bank = signal_bank.build_bank(c.J, c.E, warmed=False) \
            + warm_bank
        print(f"[stage0] eval pass over {len(c.held_keys)} held seqs ...",
              flush=True)
        results.update(eval_pass(c, full_bank, sigma, bank.bias, cfg))

    if "evict" in steps:
        dumps = results.get("evict_dumps")
        if not dumps:      # standalone evict over an earlier run's dumps
            dumps = {n: str(work / f"preds_{n}.npz")
                     for n in EVICT_DUMP_MEMBERS
                     if (work / f"preds_{n}.npz").is_file()}
        if not dumps:
            raise FileNotFoundError("no evict dumps — run the eval step "
                                    "first")
        caps = [int(x) for x in a.evict_caps.split(",")]
        results["evict"] = run_evict(dumps, caps)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
        write_table(results, Path(a.out).with_name("results_table.md"))
        print(f"[stage0] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
