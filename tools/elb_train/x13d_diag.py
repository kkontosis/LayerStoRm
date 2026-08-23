"""x13d decision-rule diagnostics (pre-registered in
studies/epm/p11-stagecprime/ARCHITECT_REVIEW.md 'Decision rule'):

  (1) per-layer ACTIVE-GAIN (f_m argmin) flips WITHIN sequences,
  (2) per-token-bucket innovation MSE spread across buckets.

One post-hoc corpus pass, no new model, no modification of the x13
engine: drives stagecprime_gpu.CPrimeState's own features()/update()
over warm+held in corpus order (the filter trajectory depends only on
the observed sel/top streams, so it reproduces the x13b run's filter
state exactly), recording gidx per position and active-gain
innovations bucketed by the 4096-bucket embedding lut.

Usage:
  python3 tools/elb_train/x13d_diag.py --shards ... --routing ...
      --device cuda:3 --workdir build/recur/p11-stagecprime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import glm_router, stage0  # noqa: E402
from elb_train.stagecprime_gpu import (CPrimeState, ExpCfg,  # noqa: E402
                                       HOLT_BETA)


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--split-seed", type=int, default=35)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigma", default="build/recur/p11-stage0/sigma.json")
    ap.add_argument("--workdir", default="build/recur/p11-stagecprime")
    ap.add_argument("--attn-buckets", type=int, default=4096)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    work = Path(a.workdir)
    c = stage0.Corpus(a.shards, a.routing, a.split_seed,
                      a.held_fraction)
    sigma = np.asarray(json.load(open(a.sigma))["sigma"], np.float32)
    bank = glm_router.RouterBank.from_npz(work / "router_bank.npz")
    bias_t = torch.as_tensor(bank.bias, device=dev)
    probe = np.load(work / "probe_full.npy")
    lut = np.load(work / "embed_lut_b154880.npz")["lut"]
    blut = np.load(work / f"embed_lut_b{a.attn_buckets}.npz")["lut"]
    cfg = ExpCfg(name="x13d_diag",
                 features=("xprev_up", "xprev_dn", "memo"),
                 members=("memo",), open_ch=True, memo_buckets=154880,
                 soft_head=True, filter_feats=True)
    st = CPrimeState(c.J, c.E, c.K, dev, probe, sigma, cfg)

    n_gain = len(st.f_m)
    flips_per_seq: list[int] = []          # summed over layers
    flips_layer = np.zeros(c.J, np.int64)  # held only
    n_pos_held = 0
    B = a.attn_buckets
    bsum = torch.zeros(B, device=dev, dtype=torch.float64)
    bcnt = torch.zeros(B, device=dev, dtype=torch.float64)

    t0 = time.time()
    for phase, keys in (("warm", c.train_keys),
                        ("held", sorted(int(s) for s in c.held_keys))):
        seqs = stage0.gather_split_cells(c, keys)
        n_done = 0
        for run_idx, skeys in stage0._by_run(seqs):
            v = c.view(run_idx)
            for key in skeys:
                s = seqs[key]
                recs = c.lookup_records(run_idx, s["seq_id"], s["pos"])
                tops_all = torch.as_tensor(
                    np.ascontiguousarray(v.top_ids[recs]), device=dev,
                    dtype=torch.int64)
                topw_all = torch.as_tensor(
                    np.ascontiguousarray(v.top_w[recs]),
                    device=dev).float()
                lg = torch.as_tensor(v.logits(recs), device=dev).float()
                sel_all = (0.5 * torch.tanh(0.5 * lg) + 0.5) \
                    + bias_t[None]
                buckets = lut[s["token"]].astype(np.int64)
                bglobal = torch.as_tensor(
                    blut[s["token"]].astype(np.int64), device=dev)
                st.reset_seq()
                prev_gidx = None
                seq_flips = 0
                for t in range(len(s["pos"])):
                    tops = tops_all[t]
                    sel = sel_all[t]
                    gidx = st.f_m.argmin(dim=0)            # [J]
                    if prev_gidx is not None:
                        d = (gidx != prev_gidx)
                        seq_flips += int(d.sum())
                        if phase == "held":
                            flips_layer += d.cpu().numpy()
                    prev_gidx = gidx
                    if phase == "held" and st.t_idx >= 1:
                        # active-gain one-step innovation vs prior
                        lv = st.f_lv[gidx, st.jj]
                        tr = st.f_tr[gidx, st.jj]
                        innov = sel - (lv + HOLT_BETA * tr)
                        mse = (innov ** 2).mean().double()
                        bsum[bglobal[t]] += mse
                        bcnt[bglobal[t]] += 1.0
                        n_pos_held += 1
                    if st.t_idx >= 1:
                        X, valid, raw = st.features(bucket=int(
                            buckets[t]))
                    else:
                        X = torch.zeros(c.J, c.E, st.F, device=dev)
                        valid = torch.zeros(c.J, c.E, device=dev,
                                            dtype=torch.bool)
                    yb = torch.zeros(c.J, c.E, device=dev,
                                     dtype=torch.bool)
                    yb.scatter_(1, tops, True)
                    st.update(tops, sel, topw_all[t], X, valid, yb,
                              int(buckets[t]))
                if phase == "held":
                    flips_per_seq.append(seq_flips)
                n_done += 1
                if n_done % 40 == 0:
                    print(f"[x13d-diag] {phase} {n_done}/{len(seqs)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            c.drop_run(run_idx)

    bs = bsum.cpu().numpy()
    bc = bcnt.cpu().numpy()
    ok = bc >= 20                          # support floor
    mse_b = bs[ok] / bc[ok]
    qs = np.percentile(mse_b, [10, 50, 90])
    out = {
        "held_positions": n_pos_held,
        "gain_grid": len(st.f_m),
        "flips": {
            "per_seq_mean": float(np.mean(flips_per_seq)),
            "per_seq_median": float(np.median(flips_per_seq)),
            "seqs_with_any_flip": int(np.sum(
                np.asarray(flips_per_seq) > 0)),
            "n_seqs": len(flips_per_seq),
            "layers_with_any_flip": int((flips_layer > 0).sum()),
            "flips_per_layer_mean": float(flips_layer.mean()),
        },
        "bucket_innovation_mse": {
            "buckets_supported": int(ok.sum()),
            "p10": float(qs[0]), "p50": float(qs[1]),
            "p90": float(qs[2]),
            "spread_p90_p10": float(qs[2] / max(qs[0], 1e-30)),
            "spread_max_median": float(mse_b.max()
                                       / max(qs[1], 1e-30)),
            "gate_2x_fires": bool(qs[2] / max(qs[0], 1e-30) >= 2.0),
        },
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(out, indent=1), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
