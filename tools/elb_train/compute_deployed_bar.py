"""EPM-5 DEPLOYED-side bar: prev-CHUNK-union coverage + hybrid coverage.

The M4 look-ahead-manifest deployed metric (NOT the within-block b0_prev
union): for each speculative BLOCK (= a verify chunk / round) N in a
sequence, at deep MoE layers L58-77, how well does a prefetch MANIFEST
cover the experts block N actually needs?

  U_cur(N,j)  = union over block N's labeled positions of labels_top_ids[.,j]
  coverage    = |manifest(N,j) ∩ U_cur(N,j)| / |U_cur(N,j)|

Two manifests, both scored on the held-out split, deep layers, chunk-
(block-)weighted:
  * PREV-CHUNK-UNION (the zero-cost BAR, ~0.509 target): manifest = the
    sequence-previous block's own union U_cur(N-1,j).
  * HYBRID (c03) MANIFEST: manifest = union over block N's positions of
    the trained hybrid's top-m predictions at layer j (uses block N's
    draft hiddens — a per-chunk forward). Target >=0.65 deep AND
    byte-cost (avg manifest size) <= 2x the prev-union's.

Run (GPU 2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python3 tools/elb_train/compute_deployed_bar.py \
      --shards /srv/models/epm5-corpus/shards --split-seed 35 \
      --hybrid build/epm5-grid/c03_hybrid_layer/model \
      --out studies/epm/epm5-corpus/bars_deployed.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elb_train import train as T, model as model_mod, dataset, glm_router  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--held-fraction", type=float, default=0.25)
    ap.add_argument("--hybrid", default="", help="hybrid model prefix (optional)")
    ap.add_argument("--deep-lo", type=int, default=58)
    ap.add_argument("--deep-hi", type=int, default=77)
    ap.add_argument("--pred-m", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    store = T.CachedShards(a.shards)
    idx = store.index
    ml = np.asarray(idx["moe_layers"], np.int32)
    deep = np.array([i for i, l in enumerate(ml)
                     if a.deep_lo <= l <= a.deep_hi])
    K = idx["topk"]; G = idx["max_gamma"]; E = idx["n_experts"]
    allk = sorted({k for sh in idx["shards"] for k in sh["seq_keys"]})
    _, held = dataset.sequence_split(allk, a.held_fraction, seed=a.split_seed)
    heset = set(held)

    # Gather held blocks: (seq_key, anchor_pos, shard, row). Per deep layer
    # store U_cur as a python set of expert ids.
    print("[deployed] gathering held block unions ...", flush=True)
    t = time.time()
    blocks = []   # dicts: seq_key, anchor, si, row, ucur[list of sets over deep]
    for si, sh in enumerate(idx["shards"]):
        if not heset & set(sh["seq_keys"]):
            continue
        z = store.shard(si)
        # anchor_pos is not in the loader's needed-key set (TD-EPM-LOADER-MEM);
        # read it directly (light — one small member).
        anchor = T.open_shard_mmap(
            Path(a.shards) / idx["shards"][si]["path"],
            keys={"anchor_pos"})["anchor_pos"]
        keep = np.nonzero(np.isin(z["seq_key"].astype(np.int64), held))[0]
        for r in keep:
            g = int(z["gamma"][r])
            lm = z["label_mask"][r, :g]
            tids = z["labels_top_ids"][r, :g]          # [g,J,K]
            ucur = []
            for jj in deep:
                s = set(int(x) for x in tids[np.nonzero(lm)[0], jj].ravel()
                        if x >= 0)
                ucur.append(s)
            blocks.append({"seq_key": int(z["seq_key"][r]),
                           "anchor": int(anchor[r]),
                           "si": int(si), "row": int(r), "ucur": ucur})
    print(f"[deployed] {len(blocks)} held blocks in {time.time()-t:.0f}s",
          flush=True)

    # order per sequence
    by_seq: dict[int, list] = {}
    for b in blocks:
        by_seq.setdefault(b["seq_key"], []).append(b)
    for v in by_seq.values():
        v.sort(key=lambda b: b["anchor"])

    # ── PREV-CHUNK-UNION bar ─────────────────────────────────────────────
    cov_sum = 0.0; cov_n = 0; prev_size_sum = 0; prev_size_n = 0
    per_layer_cov = np.zeros(len(deep)); per_layer_n = np.zeros(len(deep))
    for v in by_seq.values():
        for i in range(1, len(v)):
            cur, prev = v[i]["ucur"], v[i - 1]["ucur"]
            for di in range(len(deep)):
                uc = cur[di]
                if not uc:
                    continue
                up = prev[di]
                c = len(uc & up) / len(uc)
                cov_sum += c; cov_n += 1
                per_layer_cov[di] += c; per_layer_n[di] += 1
                prev_size_sum += len(up); prev_size_n += 1
    prev_bar = cov_sum / max(1, cov_n)
    prev_size = prev_size_sum / max(1, prev_size_n)

    result = {
        "held_seqs": len(held), "held_blocks": len(blocks),
        "deep_layers": [int(ml[i]) for i in deep],
        "prev_chunk_union_deep_cov": prev_bar,
        "prev_chunk_union_avg_size": prev_size,
        "prev_chunk_union_per_layer": (per_layer_cov /
                                       np.maximum(per_layer_n, 1)).tolist(),
    }
    print(f"[deployed] PREV-CHUNK-UNION deep coverage = {prev_bar:.4f} "
          f"(avg manifest size {prev_size:.1f} experts/layer)", flush=True)

    # ── HYBRID manifest coverage ─────────────────────────────────────────
    if a.hybrid:
        rcfg_dir = None
        # bank for selection bias / router arm
        bank = None
        try:
            import json as _j
            side = _j.load(open(str(a.hybrid) + ".sidecar.json"))
            need_bank = (side["model_cfg"].get("arm") == "router" or
                         side["model_cfg"].get("direct_head", {}).get(
                             "use_router_bias_for_selection", True))
            if need_bank:
                bank = glm_router.RouterBank.from_checkpoint(
                    "/srv/models/lukealonso/GLM-5.2-NVFP4", ml)
        except FileNotFoundError:
            pass
        model, _ = model_mod.load_model(a.hybrid, bank, device=a.device)
        model.eval()
        sb = model.selection_bias()
        bias_t = (torch.as_tensor(sb, dtype=torch.float32, device=a.device)
                  if sb is not None else None)
        eps = torch.arange(E, device=a.device, dtype=torch.float32) * 1e-6
        hcov_sum = 0.0; hcov_n = 0; hsize_sum = 0; hsize_n = 0
        hpl_cov = np.zeros(len(deep)); hpl_n = np.zeros(len(deep))
        with torch.no_grad():
            for b in blocks:
                z = store.shard(b["si"]); r = b["row"]
                g = int(z["gamma"][r])
                feats = glm_router.bf16_bits_to_f32(
                    z["features_bf16"][r, :g])[None]        # [1,g,L,H]
                x = torch.as_tensor(feats, device=a.device)
                prior = None
                if getattr(model, "prior_enabled", False):
                    prev_sh = z.get("prev_top_ids")
                    if prev_sh is not None:
                        pt = prev_sh[r, :g][None]           # [1,g,J,K]
                        tids = torch.as_tensor(np.ascontiguousarray(pt),
                                               dtype=torch.int64,
                                               device=a.device)
                        prior = torch.zeros((1, g, len(ml), E), device=a.device)
                        prior.scatter_(-1, tids.clamp(min=0),
                                       (tids >= 0).to(prior.dtype))
                logits = model(x, prior)[0]                 # [g,J,E]
                sel = torch.sigmoid(logits)
                if bias_t is not None:
                    sel = sel + bias_t
                sel = sel - eps
                pred = torch.topk(sel, a.pred_m, dim=-1).indices  # [g,J,m]
                pred = pred[:, deep, :].cpu().numpy()       # [g,deep,m]
                for di in range(len(deep)):
                    uc = b["ucur"][di]
                    if not uc:
                        continue
                    up = set(int(x) for x in pred[:, di, :].ravel())
                    hcov_sum += len(uc & up) / len(uc); hcov_n += 1
                    hpl_cov[di] += len(uc & up) / len(uc); hpl_n[di] += 1
                    hsize_sum += len(up); hsize_n += 1
        result["hybrid_deep_cov"] = hcov_sum / max(1, hcov_n)
        result["hybrid_avg_size"] = hsize_sum / max(1, hsize_n)
        result["hybrid_per_layer"] = (hpl_cov /
                                      np.maximum(hpl_n, 1)).tolist()
        result["pred_m"] = a.pred_m
        print(f"[deployed] HYBRID deep coverage = {result['hybrid_deep_cov']:.4f} "
              f"(avg manifest {result['hybrid_avg_size']:.1f} experts/layer, "
              f"m={a.pred_m}; prev-union size {prev_size:.1f}, "
              f"2x = {2*prev_size:.1f})", flush=True)

    json.dump(result, open(a.out, "w"), indent=1)
    print(f"[deployed] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
