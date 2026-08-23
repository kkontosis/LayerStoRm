"""Golden parity fixture for the C++ ExpertRidge inference module.

Runs a small-dims (J=4, E=32, K=4) but semantically COMPLETE ship3
configuration (base features + xprev/trans + memo + memo2 + ctx +
hard xsame Δ=2 + masked recur16/manifest heads + π̂ head) through the
Python engine for N seeded positions, recording per checkpoint the
raw inputs and every consumer output the C++ module must reproduce
(fp32-tolerance): X feature matrix, w_open/w_recur/w_manifest scores,
pool top-M ranking, π̂ coverage curves, and the final solved head
weights + table states. The C++ unit test replays the same inputs
and asserts closeness.

Usage: python3 tools/elb_train/export_cpp_fixture.py \
           [out.npz] (default tests/fixtures/expert_ridge_parity.npz)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train.stagecprime_gpu import CPrimeState, ExpCfg  # noqa: E402
from elb_train.stagecprime_attn import CtxState  # noqa: E402

J, E, K = 4, 32, 4
V, D_TOK = 96, 32                 # tiny frozen PCA token table
MEMO_B, MEMO2_B = 32, 64
N_POS = 100
XSAME_LAG = 2
CHECKS = (3, 40, 70, N_POS - 1)   # checkpoints straddle the t=64 refresh
POOL_M = 12                       # small-dims analog of pool 32


def main(out_path: str) -> None:
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(7)
    cfg = ExpCfg(name="fix",
                 features=("xprev_up", "xprev_dn", "memo", "memo2",
                           "ctx", "xsame"),
                 members=("memo",), open_ch=True,
                 memo_buckets=MEMO_B, memo2_buckets=MEMO2_B,
                 recur_horizons=(16,), manifest_head=True,
                 head_mask_features=("ctx", "xsame"),
                 ctx_d=48, ctx_d_out=32,
                 xsame_lag=XSAME_LAG, xsame_soft=False,
                 cal_head=True)
    probe = np.concatenate([np.zeros(12), np.ones(12),
                            np.zeros(13)]).astype(np.float32)
    sigma = np.full(J, 0.1, np.float32)
    st = CPrimeState(J, E, K, torch.device("cpu"), probe, sigma, cfg)
    emb = torch.randn(V, D_TOK, generator=g)
    ctxs = CtxState(J, E, emb, torch.device("cpu"), d=48, d_out=32)

    rec = {"J": J, "E": E, "K": K, "V": V, "d_tok": D_TOK,
           "memo_buckets": MEMO_B, "memo2_buckets": MEMO2_B,
           "xsame_lag": XSAME_LAG, "pool_m": POOL_M,
           "n_pos": N_POS, "checks": np.array(CHECKS),
           "feat_names": np.array(list(st.feat_names)),
           "pca": emb.numpy().astype(np.float32)}
    ins = {k: [] for k in ("tops", "sel", "top_w", "bucket",
                           "bucket2", "tok")}
    outs: dict = {}

    for t in range(N_POS):
        tops = torch.randint(0, E, (J, K), generator=g)
        sel = torch.rand(J, E, generator=g)
        top_w = torch.rand(J, K, generator=g)
        bucket = int(torch.randint(0, MEMO_B, (1,),
                                   generator=g).item())
        bucket2 = int(torch.randint(0, MEMO2_B, (1,),
                                    generator=g).item())
        tok = int(torch.randint(0, V, (1,), generator=g).item())
        for k, v in (("tops", tops.numpy()), ("sel", sel.numpy()),
                     ("top_w", top_w.numpy()), ("bucket", bucket),
                     ("bucket2", bucket2), ("tok", tok)):
            ins[k].append(v)
        yb = torch.zeros(J, E, dtype=torch.bool)
        yb.scatter_(1, tops, True)
        if st.t_idx >= 1:
            ctx_now = ctxs.predict(tok)
            X, valid, raw = st.features(bucket=bucket,
                                        bucket2=bucket2,
                                        ctx_now=ctx_now,
                                        sel_now=sel, tops_now=tops)
            Xa = torch.cat([X.double(),
                            torch.ones(J, E, 1,
                                       dtype=torch.float64)], 2)
            sc_open = torch.einsum("jef,jf->je", Xa, st.wo)
            sc_rec = torch.einsum("jef,jf->je", Xa, st.wrec_h[16])
            sc_man = torch.einsum("jef,jf->je", Xa, st.wman)
            in_prev = st.prev8_mask > 0
            if t in CHECKS:
                cp = f"t{t}"
                outs[f"{cp}_X"] = X.numpy().astype(np.float32)
                outs[f"{cp}_ctx_now"] = ctx_now.numpy() \
                    .astype(np.float32)
                outs[f"{cp}_sc_open"] = sc_open.numpy() \
                    .astype(np.float64)
                outs[f"{cp}_sc_rec"] = sc_rec.numpy() \
                    .astype(np.float64)
                outs[f"{cp}_sc_man"] = sc_man.numpy() \
                    .astype(np.float64)
                # pool: prev-exempt open ranking top-(POOL_M - K)
                neg = torch.finfo(torch.float32).min
                so = torch.where(~in_prev, sc_open.float(),
                                 torch.full_like(sc_open.float(),
                                                 neg))
                order = torch.argsort(so, dim=1, descending=True,
                                      stable=True)
                outs[f"{cp}_pool_open"] = order[:, : POOL_M - K] \
                    .numpy().astype(np.int64)
                outs[f"{cp}_cal_curve"] = st.cal_curve(
                    [POOL_M]).numpy().astype(np.float64)
            # π̂ update (predict-then-update; matches the engine loop)
            st.cal_update(sc_open.float(), in_prev, yb)
        else:
            X = torch.zeros(J, E, st.F)
            valid = torch.zeros(J, E, dtype=torch.bool)
        # DEPLOYMENT contract: the ctx encoder ships FROZEN (online
        # Adam is training-time only). dr=all-False maintains the
        # token ring without stepping the params, so the C++ frozen
        # forward reproduces every checkpoint exactly.
        ctxs.update(yb, torch.zeros(J, E, dtype=torch.bool), tok)
        st.update(tops, sel, top_w, X, valid, yb, bucket, bucket2)

    # final persistent state (post-run) — table/weight parity
    outs["final_wo"] = st.wo.numpy()
    outs["final_wrec16"] = st.wrec_h[16].numpy()
    outs["final_wman"] = st.wman.numpy()
    outs["final_trans"] = (st.trans * st.trans_scale).numpy() \
        .astype(np.float32)
    outs["final_xsame"] = (st.xsame * st.xsame_scale).numpy() \
        .astype(np.float32)
    outs["final_calH"] = st.calH.numpy()
    outs["final_calP"] = st.calP.numpy()
    outs["final_cal_n"] = np.float64(st.cal_n)
    # frozen encoder params (deployment form): named state_dict
    for k, v in ctxs.model.state_dict().items():
        outs[f"ctx_{k.replace('.', '_')}"] = v.numpy() \
            .astype(np.float32)

    # single-file SAFETENSORS (the C++ test consumes it through the
    # engine's own SafetensorsReader; scalars ride as 0-d -> [1])
    from elb_train.stagecprime import write_safetensors
    tensors: dict = {}
    for src in (rec, {f"in_{k}": np.array(v) for k, v in ins.items()},
                outs):
        for k, v in src.items():
            a = np.asarray(v)
            if a.dtype.kind in "US":
                a = np.frombuffer(
                    ",".join(map(str, np.atleast_1d(v).tolist()))
                    .encode(), dtype=np.uint8).copy()
                k = f"{k}_csv"
            if a.ndim == 0:
                a = a[None]
            if a.dtype == np.float16:
                a = a.astype(np.float32)
            if a.dtype == np.int32:
                a = a.astype(np.int64)
            if a.dtype == np.int16:
                a = a.astype(np.int64)
            if a.dtype == np.bool_:
                a = a.astype(np.uint8)
            tensors[k] = np.ascontiguousarray(a)
    write_safetensors(Path(out_path), tensors)
    print(f"fixture -> {out_path} "
          f"({Path(out_path).stat().st_size/1e3:.0f} KB, "
          f"{len(tensors)} tensors)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "tests/assets/expert_ridge_parity.safetensors"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    main(out)
