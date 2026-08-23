"""EPM-5 arms grid — IN-PROCESS runner (Phase C).

Loads the CachedShards + frozen router bank ONCE and trains every arm
cell against the SAME shared store and the SAME held-out split (E4),
then evaluates each cell with a NUMPY-VECTORIZED recall (the metrics
EvalAccumulator per-cell set path is impractical at 26k held blocks).
This avoids the per-cell-subprocess re-decompression of ~110 compressed
shards that run_epm4_grid would pay (TD-EPM-LOADER-MEM).

Scores each cell + the b0_prev/b0_anchor/hidden baselines on the DEPLOY
W-score (metrics.wmap_weighted_recall) and per-position recall@8 vs the
bars (b0_prev 0.302). Writes results_table.md + results.json to the
study dir.

Run (GPU 2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python3 tools/elb_train/run_epm5_grid.py \
      --base tools/elb_train/configs/epm5-base.json \
      --grid tools/elb_train/configs/epm5-arms.json \
      --study studies/epm/epm5-grid [--cells c01_direct_bce,c03_hybrid_layer]
      [--max-steps N] [--eval-batch 64]
"""
from __future__ import annotations
import argparse, copy, json, time, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from elb_train import train as T, model as model_mod, glm_router, wmap as wmap_mod  # noqa: E402
from elb_train import dataset, metrics  # noqa: E402


def _set(cfg: dict, dotted: str) -> None:
    """Apply one 'a.b.c=<json>' override (run_epm4_grid --set semantics)."""
    key, _, raw = dotted.partition("=")
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = val


def _recall_cells(pred, true):
    m = (pred[..., :, None] == true[..., None, :]) & (true[..., None, :] >= 0)
    inter = m.any(axis=-2).sum(axis=-1)
    ntrue = (true >= 0).sum(axis=-1)
    return inter, ntrue


@torch.no_grad()
def vectorized_eval(model, store, held_rows_by_shard, dims, device,
                    eval_batch=64):
    """Per-(k,j) recall@8 over the held split. GPU top-8 selection
    (sigmoid(logits)+bias, ties->lower index via a tiny index epsilon —
    the rank_experts semantics) instead of the numpy sort (the ~8 min/cell
    tax at 26k held blocks). Returns (recall_kj [G,J], ncell_kj [G,J])."""
    G, J, E = dims.max_gamma, dims.n_layers, dims.n_experts
    inter = np.zeros((G, J)); ncell = np.zeros((G, J))
    model.eval()
    sb = model.selection_bias()
    bias_t = (torch.as_tensor(sb, dtype=torch.float32, device=device)
              if sb is not None else None)          # [J,E]
    eps = torch.arange(E, device=device, dtype=torch.float32) * 1e-6
    K = store.index["topk"]
    for si, rows in held_rows_by_shard.items():
        sh = store.shard(si)
        prev_sh = sh.get("prev_top_ids")
        for b0 in range(0, len(rows), eval_batch):
            rs = rows[b0:b0 + eval_batch]
            g = max(int(sh["gamma"][r]) for r in rs)
            feats = np.zeros((len(rs), g, dims.n_taps, dims.hidden), np.float32)
            true_np = np.full((len(rs), g, J, K), -1, np.int64)
            lm_np = np.zeros((len(rs), g), bool)
            for i, r in enumerate(rs):
                gr = int(sh["gamma"][r])
                feats[i, :gr] = glm_router.bf16_bits_to_f32(
                    sh["features_bf16"][r, :gr])
                true_np[i, :gr] = sh["labels_top_ids"][r, :gr]
                lm_np[i, :gr] = sh["label_mask"][r, :gr]
            x = torch.as_tensor(feats, device=device)
            prior = None
            if getattr(model, "prior_enabled", False) and prev_sh is not None:
                pt = np.full((len(rs), g, J, K), -1, np.int64)
                for i, r in enumerate(rs):
                    gr = int(sh["gamma"][r]); pt[i, :gr] = prev_sh[r, :gr]
                tids = torch.as_tensor(pt, device=device)
                prior = torch.zeros((len(rs), g, J, E), device=device)
                prior.scatter_(-1, tids.clamp(min=0),
                               (tids >= 0).to(prior.dtype))
            logits = model(x, prior)                       # [b,g,J,E]
            sel = torch.sigmoid(logits)
            if bias_t is not None:
                sel = sel + bias_t
            sel = sel - eps                                # ties -> lower idx
            ids = torch.topk(sel, 8, dim=-1).indices       # [b,g,J,8]
            # recall on GPU: |pred ∩ true| / |true| per (b,g,J)
            true_t = torch.as_tensor(true_np, device=device)      # [b,g,J,K]
            match = (ids[..., :, None] == true_t[..., None, :]) & \
                    (true_t[..., None, :] >= 0)
            ic = match.any(-2).sum(-1).float()             # [b,g,J]
            nt = (true_t >= 0).sum(-1).float()
            lm_t = torch.as_tensor(lm_np, device=device)
            valid = lm_t[:, :, None] & (nt > 0)
            rc = torch.where(nt > 0, ic / nt.clamp(min=1),
                             torch.zeros_like(ic))
            inter += (torch.where(valid, rc, torch.zeros_like(rc))
                      .sum(0).cpu().numpy())
            ncell += valid.sum(0).cpu().numpy()
    recall_kj = np.where(ncell > 0, inter / np.maximum(ncell, 1), np.nan)
    return recall_kj, ncell


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--work", default="build/epm5-grid")
    ap.add_argument("--cells", default="")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="override optim.max_steps for every cell (0 = config)")
    ap.add_argument("--eval-batch", type=int, default=64)
    a = ap.parse_args(argv)
    base = json.load(open(a.base))
    grid = json.load(open(a.grid))
    cells = grid["cells"]
    if a.cells:
        keep = set(a.cells.split(","))
        cells = [c for c in cells if c["name"] in keep]
    study = Path(a.study); study.mkdir(parents=True, exist_ok=True)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    device = base.get("device", "cuda")

    # ── shared store + bank + split (loaded ONCE) ────────────────────────
    t0 = time.time()
    print(f"[grid] loading shared store {base['data']['shards']} ...", flush=True)
    store = T.CachedShards(base["data"]["shards"])
    dims = model_mod.Dims.from_index(store.index)
    bank = None
    rcfg = base.get("router", {})
    if rcfg.get("npz"):
        bank = glm_router.RouterBank.from_npz(rcfg["npz"])
    elif rcfg.get("checkpoint_dir"):
        bank = glm_router.RouterBank.from_checkpoint(
            rcfg["checkpoint_dir"], dims.moe_layers)
    allk = sorted({k for sh in store.index["shards"] for k in sh["seq_keys"]})
    train_keys, held_keys = dataset.sequence_split(
        allk, base["data"]["held_out_fraction"],
        seed=int(base["data"]["split_seed"]))
    print(f"[grid] split: {len(train_keys)} train / {len(held_keys)} held seqs",
          flush=True)
    held_rows = store.rows(held_keys)
    held_by_shard: dict[int, list] = {}
    for si, r in held_rows:
        held_by_shard.setdefault(si, []).append(r)
    ml = np.asarray(store.index["moe_layers"], np.int32)
    deep = np.array([i for i, l in enumerate(ml) if 58 <= l <= 77])
    # deployment W (same for all cells): the DIRECTIVE W on this corpus
    w_deploy = wmap_mod.build_wmap(
        wmap_mod.WMapSpec.from_dict(base["wmap"]), dims.max_gamma, ml)

    def score(recall_kj, ncell):
        r8 = float(np.nansum(recall_kj * ncell) / np.nansum(ncell))
        pp = [float(np.nansum(recall_kj[k] * ncell[k]) /
                    max(1, np.nansum(ncell[k]))) for k in range(dims.max_gamma)]
        wscore = float(np.nansum(recall_kj * ncell * w_deploy) /
                       np.nansum(ncell * w_deploy))
        deep_r8 = float(np.nansum(recall_kj[:, deep] * ncell[:, deep]) /
                        np.nansum(ncell[:, deep]))
        return {"recall8": r8, "w_score": wscore, "deep_recall8": deep_r8,
                "per_position": pp}

    results = {}
    print(f"[grid] store+bank ready in {time.time()-t0:.0f}s\n", flush=True)

    for cell in cells:
        name = cell["name"]
        cfg = copy.deepcopy(base)
        for ov in cell.get("overrides", []):
            _set(cfg, ov)
        if a.max_steps > 0:
            cfg["optim"]["max_steps"] = a.max_steps
        ct = time.time()
        tr = T.Trainer(cfg, out_dir=work / name, router_bank=bank, store=store)
        tr.train()
        rkj, nc = vectorized_eval(tr.model, store, held_by_shard, dims,
                                  device, a.eval_batch)
        s = score(rkj, nc)
        s["note"] = cell.get("note", "")
        s["seconds"] = round(time.time() - ct)
        results[name] = s
        model_mod.save_model(tr.model, work / name / "model")
        print(f"[grid] {name}: recall@8 {s['recall8']:.4f} "
              f"W {s['w_score']:.4f} deep {s['deep_recall8']:.4f} "
              f"({s['seconds']}s)", flush=True)
        del tr
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        json.dump(results, open(study / "results.json", "w"), indent=1)

    # table
    lines = ["| cell | recall@8 | W-score | deep-r@8 | note |",
             "|---|---|---|---|---|"]
    for n in sorted(results, key=lambda n: -results[n]["w_score"]):
        r = results[n]
        lines.append(f"| {n} | {r['recall8']:.4f} | {r['w_score']:.4f} "
                     f"| {r['deep_recall8']:.4f} | {r['note']} |")
    lines.append(f"| *b0_prev (bar)* | 0.3043 | — | — | temporal locality |")
    table = "\n".join(lines)
    open(study / "results_table.md", "w").write(table + "\n")
    print("\n" + table, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
