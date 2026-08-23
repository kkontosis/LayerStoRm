"""GPU engine for the P11.c online mixture (torch, single device).

The CPU engine loops layers with variable-size candidate sets; here
everything is DENSE over the expert axis with a validity mask
(candidates = trail-64 union, invalid experts ranked to -inf), so one
position = a fixed sequence of batched [J, E] tensor ops. Sequences are
processed serially (the model is online); per-sequence H2D of
tops/top_w/logits, per-sequence D2H of the evict rows, all accumulators
live on the device — no per-position syncs.

Parity contract with stagec._stagec_eval_seqs: same members, same
rank-normalization (stable sorts tie-break by expert id exactly like
the CPU's stable argsort over ascending candidate ids), same Hedge/
ridge updates, same metric definitions; returns the same partial dict,
so run_stagec's merge/verdict path is shared. Tie-float differences
bound verdict deltas at ~1e-3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import stage0, stageb
from .stageb import EVICT_PROBE_TOPK, MANIFEST_DS, POOL_KS
from .stageb_features import FEATURES, RECENCY_CAP
from .stagec import (BANK, HEDGE_ETA, RLS_DECAY, RLS_LAMBDA,
                     RLS_REFRESH)

VARIANTS = ("combined", "probe_frozen", "rls_online", "trans_ema")
F = len(FEATURES)


class GpuState:
    def __init__(self, J: int, E: int, K: int, dev: torch.device,
                 probe_packed: np.ndarray, sigma: np.ndarray):
        self.J, self.E, self.K, self.dev = J, E, K, dev
        f32 = dict(device=dev, dtype=torch.float32)
        self.t16 = torch.zeros(J, E, **f32)
        self.t64 = torch.zeros(J, E, **f32)
        self.ring16: list = []
        self.ring64: list = []
        self.trans = torch.zeros(J, E, E, **f32)
        self.trans_scale = 1.0
        self.prev8: torch.Tensor | None = None       # [J, K] i64
        self.prev8_mask = torch.zeros(J, E, **f32)
        self.ema2 = torch.zeros(J, E, **f32)
        self.ema8 = torch.zeros(J, E, **f32)
        self.n2 = 0.0
        self.n8 = 0.0
        self.d2 = 0.5 ** (1.0 / 2.0)
        self.d8 = 0.5 ** (1.0 / 8.0)
        self.g_prev = torch.zeros(J, E, **f32)
        self.g_prev2 = torch.zeros(J, E, **f32)
        self.cut_prev = torch.zeros(J, **f32)
        self.last_seen = torch.full((J, E), -1, device=dev,
                                    dtype=torch.int64)
        self.t_idx = 0
        # ridge (f64) + hedge
        self.A = torch.zeros(J, F + 1, F + 1, device=dev,
                             dtype=torch.float64)
        self.bvec = torch.zeros(J, F + 1, device=dev, dtype=torch.float64)
        self.wr = torch.zeros(J, F + 1, device=dev, dtype=torch.float64)
        self._since = 0
        self.hw = torch.full((J, len(BANK)), 1.0 / len(BANK),
                             device=dev, dtype=torch.float64)
        # probe params
        d = F
        self.mu = torch.as_tensor(probe_packed[:d], **f32)
        self.sd = torch.as_tensor(probe_packed[d:2 * d], **f32)
        self.pw = torch.as_tensor(probe_packed[2 * d:], **f32)  # [F+1]
        self.sigma = torch.as_tensor(sigma, **f32)              # [J]
        self.jj = torch.arange(J, device=dev)
        self.layer_f = (self.jj.float() / max(1, J - 1))[:, None] \
            .expand(J, E)

    def reset_seq(self):
        self.t16.zero_()
        self.t64.zero_()
        self.ring16.clear()
        self.ring64.clear()
        self.prev8 = None
        self.prev8_mask.zero_()
        self.ema2.zero_()
        self.ema8.zero_()
        self.n2 = self.n8 = 0.0
        self.g_prev.zero_()
        self.g_prev2.zero_()
        self.cut_prev.zero_()
        self.last_seen.fill_(-1)
        self.t_idx = 0

    # ── features + member scores (dense, PRE-update state) ──────────────

    def features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(X [J,E,F], valid [J,E] bool, trans_scores [J,E])."""
        J, E = self.J, self.E
        valid = self.t64 > 0
        if self.prev8 is None:
            tp = torch.zeros(J, E, device=self.dev)
        else:
            tp = self.trans[self.jj[:, None], self.prev8].sum(dim=1) \
                * self.trans_scale
        rec = torch.where(self.last_seen >= 0,
                          (self.t_idx - self.last_seen).float(),
                          torch.full_like(self.t16, RECENCY_CAP))
        rec = rec.clamp(max=RECENCY_CAP)
        # trans_rank among valid (0 = best), CPU orientation: invalid
        # entries (-inf after negation) occupy the lowest ranks; subtract
        # their count so valid ranks start at 0 like the CPU cand array.
        tpm = torch.where(valid, tp, torch.full_like(tp, float("inf")))
        ranks = torch.argsort(
            torch.argsort(-tpm, dim=1, stable=True), dim=1,
            stable=True).float()
        n_inv = (~valid).sum(dim=1, keepdim=True).float()
        nval = valid.sum(dim=1, keepdim=True).float()
        trank = (ranks - n_inv) / (nval - 1).clamp(min=1)
        e2 = self.ema2 / self.n2 if self.n2 > 0 else self.ema2
        e8 = self.ema8 / self.n8 if self.n8 > 0 else self.ema8
        X = torch.stack([
            self.prev8_mask, self.t16, self.t64, rec,
            tp, trank, e2, e8, self.g_prev,
            self.g_prev - self.cut_prev[:, None],
            self.g_prev - self.g_prev2, self.layer_f], dim=2)
        return X, valid, tp

    def member_scores(self, X: torch.Tensor, valid: torch.Tensor,
                      tp: torch.Tensor) -> torch.Tensor:
        """[6, J, E] raw member scores (rank-normalized by caller)."""
        z = (X - self.mu) / self.sd
        probe = torch.sigmoid(
            torch.einsum("jef,f->je", z, self.pw[:-1]) + self.pw[-1])
        Xa = torch.cat([X.double(),
                        torch.ones(self.J, self.E, 1, device=self.dev,
                                   dtype=torch.float64)], dim=2)
        rls = torch.einsum("jef,jf->je", Xa, self.wr).float()
        return torch.stack([self.prev8_mask, tp, X[:, :, 6],
                            self.t16, probe, rls])

    # ── updates (reveal position t) ─────────────────────────────────────

    def update(self, tops: torch.Tensor, sel: torch.Tensor,
               X: torch.Tensor, valid: torch.Tensor,
               yb: torch.Tensor) -> None:
        """tops [J,K] i64, sel [J,E] f32, X/valid from features(),
        yb [J,E] bool (true top-K mask)."""
        J, E, K = self.J, self.E, self.K
        ones = torch.ones(J, K, device=self.dev)
        # ridge stats (candidates only)
        vf = valid.double()
        Xa = torch.cat([X.double(),
                        torch.ones(J, E, 1, device=self.dev,
                                   dtype=torch.float64)], dim=2) \
            * vf[:, :, None]
        self.A.mul_(RLS_DECAY).add_(
            torch.einsum("jef,jeg->jfg", Xa, Xa))
        self.bvec.mul_(RLS_DECAY).add_(
            torch.einsum("jef,je->jf", Xa, yb.double() * vf))
        self._since += 1
        if self._since >= RLS_REFRESH:
            eye = torch.eye(F + 1, device=self.dev,
                            dtype=torch.float64) * RLS_LAMBDA
            self.wr = torch.linalg.solve(self.A + eye,
                                         self.bvec.unsqueeze(2)) \
                .squeeze(2)
            self._since = 0
        # transition table
        if self.prev8 is not None:
            self.trans_scale *= 0.99
            w_inc = 1.0 / self.trans_scale
            flat = (self.jj[:, None, None] * E * E
                    + self.prev8[:, :, None] * E
                    + tops[:, None, :]).reshape(-1)
            self.trans.view(-1).scatter_add_(
                0, flat, torch.full_like(flat, w_inc,
                                         dtype=torch.float32))
            if self.trans_scale < 1e-20:
                self.trans.mul_(self.trans_scale)
                self.trans_scale = 1.0
        # trailing windows
        for ring, counts, hor in ((self.ring16, self.t16, 16),
                                  (self.ring64, self.t64, 64)):
            if len(ring) >= hor:
                old = ring.pop(0)
                counts.scatter_add_(1, old, -ones)
            ring.append(tops)
            counts.scatter_add_(1, tops, ones)
        # EMAs, prev, scores
        self.ema2.mul_(self.d2).add_(sel, alpha=1 - self.d2)
        self.ema8.mul_(self.d8).add_(sel, alpha=1 - self.d8)
        self.n2 = self.n2 * self.d2 + (1 - self.d2)
        self.n8 = self.n8 * self.d8 + (1 - self.d8)
        self.prev8 = tops
        self.prev8_mask.zero_().scatter_(1, tops, 1.0)
        self.g_prev2.copy_(self.g_prev)
        self.g_prev.copy_(sel)
        self.cut_prev.copy_(torch.topk(sel, K, dim=1).values[:, -1])
        self.last_seen.scatter_(
            1, tops, torch.full_like(tops, self.t_idx))
        self.t_idx += 1


def _rank_norm_dense(sc: torch.Tensor, valid: torch.Tensor
                     ) -> torch.Tensor:
    """[.., J, E] scores -> [.., J, E] rank-normalized among valid
    (invalid -> 0). CPU parity: value = ascending rank / (n_valid-1)."""
    neg_inf = torch.finfo(sc.dtype).min
    scm = torch.where(valid, sc, torch.full_like(sc, neg_inf))
    ranks = torch.argsort(torch.argsort(scm, dim=-1, stable=True),
                          dim=-1, stable=True).float()
    n_inv = (~valid).sum(dim=-1, keepdim=True).float()
    nval = valid.sum(dim=-1, keepdim=True).float()
    rn = (ranks - n_inv) / (nval - 1).clamp(min=1)
    return torch.where(valid, rn, torch.zeros_like(rn))


def gpu_eval_seqs(work: Path, c: stage0.Corpus, sigma, bias, held_keys,
                  evict_keys: set[int], deep_lo: int, deep_hi: int,
                  warm_keys, device: str = "cuda:0",
                  evict_scorer: str = "rls_online") -> dict:
    dev = torch.device(device)
    probe_packed = np.load(work / "probe_full.npy")
    st = GpuState(c.J, c.E, c.K, dev, probe_packed, sigma)
    deep_np = np.array([i for i, l in enumerate(c.moe_layers)
                        if deep_lo <= l <= deep_hi])
    deep = torch.as_tensor(deep_np, device=dev)
    bias_t = torch.as_tensor(bias, device=dev)
    sig_t = st.sigma
    n_ks = len(POOL_KS)
    ks = torch.as_tensor(POOL_KS, device=dev)

    pool_hit = torch.zeros(len(VARIANTS), n_ks, device=dev,
                           dtype=torch.int64)
    addr_hit = torch.zeros(len(VARIANTS), n_ks, device=dev,
                           dtype=torch.int64)
    pool_true = 0
    addr_n = torch.zeros((), device=dev, dtype=torch.int64)
    man = torch.zeros(len(MANIFEST_DS), 2, device=dev,
                      dtype=torch.float64)
    man_free = {alt: [0.0, 0.0, 0] for alt in ("prev_union", "two_chunk")}
    mf_acc = torch.zeros(2, 3, device=dev, dtype=torch.float64)
    dump: dict[str, list] = {k: [] for k in
                             ("seq", "pos", "layer", "expert", "prob",
                              "label", "occ_prev", "nextdist")}
    collect = {"on": False}
    prev_u: dict[int, list] = {}

    def process_seq(key: int, s: dict, v_view) -> None:
        recs = c.lookup_records(s["run_idx"], s["seq_id"], s["pos"])
        tops_np = np.ascontiguousarray(v_view.top_ids[recs])
        logits_np = v_view.logits(recs)
        T = len(s["pos"])
        tops_all = torch.as_tensor(tops_np, device=dev,
                                   dtype=torch.int64)
        # stable_sigmoid = 0.5*tanh(0.5 x)+0.5 (engine formulation)
        lg = torch.as_tensor(logits_np, device=dev).float()
        sel_all = (0.5 * torch.tanh(0.5 * lg) + 0.5) + bias_t[None]
        chunk = s["chunk"]
        st.reset_seq()
        is_evict = collect["on"] and key in evict_keys \
            and s["recur"] is not None
        prob_rows: list = []
        prob_meta: list = []
        for ci, blk in enumerate(s["blocks"]):
            # ── manifest at chunk start (pre-chunk state) ──
            frecs = c.lookup_records(s["run_idx"], s["seq_id"],
                                     blk["full_pos"])
            ft = torch.as_tensor(
                np.ascontiguousarray(v_view.top_ids[frecs]),
                device=dev, dtype=torch.int64)[:, deep]
            ucur = torch.zeros(len(deep_np), c.E, device=dev,
                               dtype=torch.bool)
            ucur.scatter_(1, ft.permute(1, 0, 2)
                          .reshape(len(deep_np), -1), True)
            state = prev_u.setdefault(key, [None, None])
            u1, u2 = state[0], state[1]
            if collect["on"] and u1 is not None and st.t_idx >= 1:
                un = ucur.sum(dim=1)
                ok = un > 0
                unc = un.clamp(min=1).double()
                cov1 = ((u1 & ucur).sum(dim=1).double() / unc)[ok]
                u12 = (u1 | u2) if u2 is not None else u1
                cov2 = ((u12 & ucur).sum(dim=1).double() / unc)[ok]
                mf_acc[0, 0] += cov1.sum()
                mf_acc[0, 1] += u1[ok].sum()
                mf_acc[0, 2] += ok.sum()
                mf_acc[1, 0] += cov2.sum()
                mf_acc[1, 1] += u12[ok].sum()
                mf_acc[1, 2] += ok.sum()
                X, valid, tp = st.features()
                raw = st.member_scores(X, valid, tp)
                norm = _rank_norm_dense(raw, valid[None])
                comb = torch.einsum("jn,nje->je", st.hw.float(), norm)
                combd = comb[deep]
                validd = valid[deep]
                resid = validd & ~u1
                neg_inf = torch.finfo(torch.float32).min
                scm = torch.where(resid, combd,
                                  torch.full_like(combd, neg_inf))
                order = torch.argsort(scm, dim=1, descending=True,
                                      stable=True)
                for di, dsize in enumerate(MANIFEST_DS):
                    add = order[:, :dsize]
                    m = u1.clone()
                    m.scatter_(1, add, resid.gather(1, add))
                    cov = ((m & ucur).sum(dim=1).double() / unc)[ok]
                    man[di, 0] += cov.sum()
                    man[di, 1] += ok.sum()
            state[1] = state[0]
            state[0] = ucur
            # ── positions of this chunk ──
            for t in np.nonzero(chunk == ci)[0]:
                tops = tops_all[t]
                sel = sel_all[t]
                if st.t_idx >= 1:
                    X, valid, tp = st.features()
                    raw = st.member_scores(X, valid, tp)
                    norm = _rank_norm_dense(raw, valid[None])
                    comb = torch.einsum("jn,nje->je", st.hw.float(),
                                        norm)
                    yb = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                    yb.scatter_(1, tops, True)
                    in_prev = st.prev8_mask > 0
                    cut9 = torch.topk(sel, c.K + 1,
                                      dim=1).values[:, -1]
                    bnd = yb & ((sel - cut9[:, None])
                                < sig_t[:, None])
                    addressable = yb & ~in_prev & ~bnd
                    if collect["on"]:
                        nonlocal pool_true
                        pool_true += c.K * c.J
                        addr_n.add_(addressable.sum())
                        var_scores = torch.stack(
                            [comb, norm[4], norm[5], norm[1]])
                        resid = (valid & ~in_prev)[None] \
                            .expand(4, -1, -1)
                        neg_inf = torch.finfo(torch.float32).min
                        scm = torch.where(
                            resid, var_scores,
                            torch.full_like(var_scores, neg_inf))
                        order = torch.argsort(scm, dim=2,
                                              descending=True,
                                              stable=True)
                        ybx = yb[None].expand(4, -1, -1) \
                            .gather(2, order) & resid.gather(2, order)
                        adx = addressable[None].expand(4, -1, -1) \
                            .gather(2, order) & resid.gather(2, order)
                        cy = ybx.long().cumsum(dim=2)
                        ca = adx.long().cumsum(dim=2)
                        prev_hits = (yb & in_prev).sum()
                        sel_k = cy[:, :, ks - 1].sum(dim=1)   # [4,n_ks]
                        pool_hit.add_(sel_k + prev_hits)
                        addr_hit.add_(ca[:, :, ks - 1].sum(dim=1))
                        if is_evict:
                            esc = {"combined": comb,
                                   "probe_frozen": norm[4],
                                   "rls_online": norm[5]}[evict_scorer]
                            csc = torch.where(
                                (valid & ~in_prev), esc,
                                torch.full_like(esc, neg_inf))
                            pidx = torch.topk(
                                csc, EVICT_PROBE_TOPK, dim=1).indices
                            prot = torch.zeros_like(yb)
                            prot.scatter_(1, pidx, True)
                            prot &= valid & ~in_prev
                            prot |= in_prev
                            prob_rows.append(
                                prot.gather(1, tops).float())
                            prob_meta.append(t)
                    # hedge update
                    tcand = (yb & valid).sum(dim=1)          # [J]
                    m8 = torch.topk(
                        torch.where(valid[None].expand(6, -1, -1), raw,
                                    torch.full_like(
                                        raw,
                                        torch.finfo(torch.float32).min)),
                        8, dim=2).indices
                    hits8 = yb[None].expand(6, -1, -1) \
                        .gather(2, m8).sum(dim=2).double()
                    denom = torch.minimum(
                        torch.full_like(tcand, 8), tcand) \
                        .clamp(min=1).double()
                    losses = 1.0 - hits8 / denom[None]
                    active = (tcand > 0).double()
                    st.hw *= torch.exp(-HEDGE_ETA * losses.t()
                                       * active[:, None])
                    st.hw /= st.hw.sum(dim=1, keepdim=True)
                    st.update(tops, sel, X, valid, yb)
                else:
                    # first position: state prime only
                    X = torch.zeros(c.J, c.E, F, device=dev)
                    valid = torch.zeros(c.J, c.E, device=dev,
                                        dtype=torch.bool)
                    yb = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                    yb.scatter_(1, tops, True)
                    st.update(tops, sel, X, valid, yb)
        if is_evict and prob_rows:
            probs = torch.stack(prob_rows).cpu().numpy()  # [Te, J, K]
            for i, t in enumerate(prob_meta):
                for j in range(c.J):
                    dump["seq"].append(np.full(c.K, key, np.int64))
                    dump["pos"].append(
                        np.full(c.K, s["pos"][t], np.int32))
                    dump["layer"].append(np.full(c.K, j, np.int16))
                    dump["expert"].append(
                        tops_np[t, j].astype(np.int16))
                    dump["prob"].append(probs[i, j].astype(np.float32))
                    dump["label"].append(s["recur"]["label"][t, j])
                    dump["occ_prev"].append(
                        s["recur"]["occ_prev"][t, j])
                    dump["nextdist"].append(
                        s["recur"]["nextdist"][t, j])

    import time
    for phase, keys in (("warm", warm_keys), ("held", held_keys)):
        if not len(keys):
            continue
        collect["on"] = phase == "held"
        seqs = stage0.gather_split_cells(
            c, keys, recur_keys=evict_keys if phase == "held"
            else frozenset())
        n_done = 0
        t0 = time.time()
        for run_idx, skeys in stage0._by_run(seqs):
            v = c.view(run_idx)
            for key in skeys:
                process_seq(key, seqs[key], v)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"[stagec-gpu] {phase} {n_done}/{len(seqs)} "
                          f"seqs ({time.time()-t0:.0f}s)", flush=True)
            c.drop_run(run_idx)

    torch.cuda.synchronize(dev)
    dump_path = None
    if dump["seq"]:
        arrs = {k: np.concatenate(v) for k, v in dump.items()}
        p = work / "preds_stagec_combined.npz"
        np.savez_compressed(p, **arrs)
        dump_path = str(p)
    ph = pool_hit.cpu().numpy()
    ah = addr_hit.cpu().numpy()
    mn = man.cpu().numpy()
    mf = mf_acc.cpu().numpy()
    ml = c.moe_layers
    deep_mask = (ml >= deep_lo) & (ml <= deep_hi)
    hw = st.hw.cpu().numpy()
    return {
        "pool_hit": {f"{v}|{k}": int(ph[i, ki])
                     for i, v in enumerate(VARIANTS)
                     for ki, k in enumerate(POOL_KS)},
        "addr_hit": {f"{v}|{k}": int(ah[i, ki])
                     for i, v in enumerate(VARIANTS)
                     for ki, k in enumerate(POOL_KS)},
        "pool_true": pool_true,
        "addr_n": int(addr_n.item()),
        "man": {str(d): [float(mn[i, 0]), int(mn[i, 1])]
                for i, d in enumerate(MANIFEST_DS)},
        "man_free": {"prev_union": [float(mf[0, 0]), float(mf[0, 1]),
                                    int(mf[0, 2])],
                     "two_chunk": [float(mf[1, 0]), float(mf[1, 1]),
                                   int(mf[1, 2])]},
        "dump_path": dump_path,
        "hedge_w_mean": hw.mean(axis=0).tolist(),
        "hedge_w_deep": hw[deep_mask].mean(axis=0).tolist(),
        "variants": list(VARIANTS),
        "evict_scorer": evict_scorer,
        "ridge_w": st.wr.cpu().numpy().tolist(),   # [J, F+1] — the
        # deployable ridge-only evictor weights (features + bias)
    }
