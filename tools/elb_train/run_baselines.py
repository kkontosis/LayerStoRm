"""EPM-2 baseline report entrypoint (experiment #0 — gates EPM-3 scope).

Runs B0/B1/B2 on an assembled shard directory and writes the study-dir
report: per-baseline metric artifacts (<name>.npz/.json — the EPM-3 input
contract, see metrics.py docstring), fidelity heatmap PNGs (skipped without
matplotlib), the best-tap table, corpus/coverage stats, and summary.md.

Usage:
  python3 tools/elb_train/run_baselines.py \
      --shards <assembled-shard-dir> --out <study-dir> \
      [--dump-root <raw-dump-corpus-root>]        # enables B0
      [--router-dir <hf-checkpoint-dir>]          # B1/B2 real routers, or
      [--router-npz <RouterBank .npz>]            #   synthetic routers
      [--held-out-fraction 0.25] [--seed 0] [--ridge-lambda 1e-6]
      [--synthetic]   # generate a self-contained synthetic corpus first
                      # (ignores --shards/--dump-root/--router-*)

All numbers reported on the HELD-OUT split (train split only feeds B1 tap
selection + B2 ridge fitting); splits are sequence-level (INV-EPM-DATA).
Metrics are acceptance-conditioned by construction (label_mask = verified
prefix; see metrics.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # `python3 tools/elb_train/run_baselines.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from elb_train import baselines, dataset, glm_router, metrics
    from elb_train import synth_corpus
else:
    from . import baselines, dataset, glm_router, metrics, synth_corpus


def _fmt_pct(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v:.4f}"


def run_report(shards: Path, out: Path, *, dump_root: Path | None,
               bank: glm_router.RouterBank | None,
               held_out_fraction: float, seed: int, ridge_lambda: float,
               provenance: str, in_sample: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    index = dataset.load_index(shards)
    stats = dataset.dataset_stats(shards)
    all_keys = sorted({k for sh in index["shards"] for k in sh["seq_keys"]})
    results: dict[str, dict] = {}
    notes: list[str] = []
    if in_sample:
        # Degenerate-corpus escape hatch (e.g. a single-sequence smoke
        # collection): selection and evaluation share ALL sequences.  The
        # numbers are IN-SAMPLE — never compare them against held-out runs.
        train = held = list(all_keys)
        notes.append("IN-SAMPLE MODE: train == eval == all sequences "
                     "(corpus too small for a sequence-level split; "
                     "INV-EPM-DATA held-out discipline NOT satisfied — "
                     "B1 tap selection and especially the B2 ridge probe "
                     "overfit freely; numbers are upper bounds).")
    else:
        train, held = dataset.sequence_split(all_keys, held_out_fraction,
                                             seed=seed)
        if not held or not train:
            raise SystemExit(f"degenerate split: {len(train)} train / "
                             f"{len(held)} held-out sequences — need both "
                             f"(use --in-sample for smoke corpora)")
    moe_layers = np.asarray(index["moe_layers"], np.int32)

    # ── B0 ────────────────────────────────────────────────────────────────
    if dump_root is not None:
        maps = baselines.load_routing_maps(dump_root)
        for variant in ("anchor", "prev"):
            name = f"b0_{variant}"
            results[name] = baselines.run_b0(shards, maps, held, variant)
    else:
        notes.append("B0 skipped: no --dump-root (needs raw epm_routing.bin"
                     " for previous-token records).")

    # ── B1 ────────────────────────────────────────────────────────────────
    sweep_train = sweep_held = None
    best_taps = prior_taps = None
    if bank is not None:
        sweep_train = baselines.b1_tap_sweep(shards, bank, train)
        sweep_held = baselines.b1_tap_sweep(shards, bank, held)
        best_taps = baselines.select_best_taps(sweep_train)
        prior_taps = glm_router.aux_prior_tap(moe_layers)
        results["b1_best_tap"] = baselines.run_b1(shards, bank, held,
                                                  best_taps)
        results["b1_prior_tap"] = baselines.run_b1(shards, bank, held,
                                                   prior_taps)
        np.savez(out / "b1_tap_sweep.npz",
                 train_recall=sweep_train["recall_table"],
                 train_n=sweep_train["n_table"],
                 heldout_recall=sweep_held["recall_table"],
                 heldout_n=sweep_held["n_table"],
                 best_taps=best_taps, prior_taps=prior_taps,
                 moe_layers=moe_layers)
    else:
        notes.append("B1/B2 skipped: no router weights (--router-dir / "
                     "--router-npz).")

    # ── B2 ────────────────────────────────────────────────────────────────
    if bank is not None:
        probe = baselines.train_b2(shards, train, best_taps,
                                   ridge_lambda=ridge_lambda)
        results["b2_linear_probe"] = baselines.run_b2(shards, probe, bank,
                                                      held)
        notes.append(f"B2: direct {index['hidden']}->{index['n_experts']}"
                     f"-logit ridge probe, input tap = B1 train-best per "
                     f"layer, lambda={ridge_lambda} (relative), n_train="
                     f"{probe['n_train']} labeled positions.")

    e = index["n_experts"]
    notes.append(f"Chance level (random {index['topk']}-of-{e} predictor): "
                 f"recall@8={8 / e:.4f}, recall@16={16 / e:.4f}, "
                 f"recall@32={32 / e:.4f}.")

    # ── artifacts ─────────────────────────────────────────────────────────
    meta = {"provenance": provenance, "shards": str(shards),
            "seed": seed, "held_out_fraction": held_out_fraction,
            "in_sample": in_sample,
            "train_sequences": len(train), "held_out_sequences": len(held)}
    png_ok = None
    for name, res in results.items():
        metrics.save_result(res, out / name, meta=meta)
        r8 = res["recall"][8]
        ok = metrics.save_heatmap_png(
            r8["mean"], r8["n"], out / f"{name}_recall8.png",
            f"{name} top-8 set recall (held-out)", moe_layers)
        png_ok = ok if png_ok is None else (png_ok and ok)
    if png_ok is False:
        notes.append("Heatmap PNGs skipped: matplotlib not installed "
                     "(npz arrays carry the same data).")

    # ── summary.md ────────────────────────────────────────────────────────
    lines = [
        f"# EPM-2 baseline report — {provenance}", "",
        "Experiment #0 (PLAN.md Phase 29 EPM-2): zero-training baselines "
        "through the coverage-aware eval harness. All metrics on the "
        "HELD-OUT sequence split; every value carries its cell count n "
        "(labels exist only on the verified prefix — "
        "TD-EPM-REJECTED-LABELS — so these ARE the acceptance-conditioned "
        "metrics).", "",
        "## Corpus", "",
        f"- blocks: {stats['num_blocks']}  sequences: "
        f"{stats['num_sequences']}  max_gamma: {stats['max_gamma']}",
        f"- MoE layers: {len(index['moe_layers'])} "
        f"({index['moe_layers'][0]}..{index['moe_layers'][-1]}), experts: "
        f"{index['n_experts']}, topk: {index['topk']}, draft taps: "
        f"{index['n_draft_layers']}, hidden: {index['hidden']}",
        f"- per-position label counts: {stats['position_label_counts']}",
        f"- per-position label coverage: "
        f"{[round(float(c), 4) for c in stats['position_label_coverage']]}",
        (f"- split: IN-SAMPLE (train == eval == {len(all_keys)} sequences"
         f" — see Notes)" if in_sample else
         f"- split: {len(train)} train / {len(held)} held-out sequences "
         f"(seed {seed}, INV-EPM-DATA sequence-level)"), "",
        f"## Fidelity ({'IN-SAMPLE' if in_sample else 'held-out'})", "",
        metrics.summary_table(results),
    ]
    if sweep_train is not None:
        eval_label = "IN-SAMPLE" if in_sample else "held-out"
        lines += ["## B1 tap sweep — best tap per layer (train-selected)",
                  "",
                  "| layer | best tap | prior tap | train r@8 best | "
                  f"train r@8 prior | {eval_label} r@8 best |",
                  "|---|---|---|---|---|---|"]
        tr = sweep_train["recall_table"]
        hr = sweep_held["recall_table"]
        for jj, layer in enumerate(moe_layers):
            bt, pt = int(best_taps[jj]), int(prior_taps[jj])
            lines.append(
                f"| {int(layer)} | {bt} | {pt} | {_fmt_pct(tr[bt, jj])} | "
                f"{_fmt_pct(tr[pt, jj])} | {_fmt_pct(hr[bt, jj])} |")
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["## Artifacts", "",
              "- `<baseline>.npz/.json` — per-(k, j) heatmaps + aggregates "
              "(contract: tools/elb_train/metrics.py docstring)",
              "- `b1_tap_sweep.npz` — full tap × layer recall tables "
              "(train + held-out) and the tap selections", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "report_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "corpus_stats": stats, "notes": notes},
                  f, indent=1)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dump-root", type=Path, default=None)
    ap.add_argument("--router-dir", type=Path, default=None)
    ap.add_argument("--router-npz", type=Path, default=None)
    ap.add_argument("--held-out-fraction", type=float, default=0.25)
    ap.add_argument("--in-sample", action="store_true",
                    help="degenerate corpora only: train == eval == all "
                         "sequences (numbers are in-sample upper bounds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge-lambda", type=float, default=1e-6)
    ap.add_argument("--provenance", default="")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate a synthetic corpus under --out and run "
                         "the full report on it (no real data needed)")
    args = ap.parse_args(argv)

    if args.synthetic:
        work = args.out / "synthetic-corpus"
        truth = synth_corpus.generate_synthetic_run(
            work / "dump" / "run_1", n_seqs=24, blocks_per_seq=8,
            gamma=4, hidden=24, moe_layers=(3, 4, 5, 10, 11, 12),
            n_experts=32, rho=0.9, tap_noise=0.05, seed=args.seed)
        dataset.assemble_shards(work / "dump", work / "shards")
        bank = truth["routers"]
        bank.save_npz(work / "routers.npz")
        shards = work / "shards"
        dump_root = work / "dump"
        provenance = (args.provenance or
                      "SYNTHETIC corpus (known generating process; real "
                      "corpus pending TD-EPM-CORPUS)")
    else:
        if args.shards is None:
            ap.error("--shards required (or use --synthetic)")
        shards = args.shards
        dump_root = args.dump_root
        bank = None
        if args.router_npz is not None:
            bank = glm_router.RouterBank.from_npz(args.router_npz)
        elif args.router_dir is not None:
            index = dataset.load_index(shards)
            bank = glm_router.RouterBank.from_checkpoint(
                args.router_dir, index["moe_layers"])
        provenance = args.provenance or f"shards: {shards}"

    run_report(shards, args.out, dump_root=dump_root, bank=bank,
               held_out_fraction=args.held_out_fraction, seed=args.seed,
               ridge_lambda=args.ridge_lambda, provenance=provenance,
               in_sample=args.in_sample)
    print(f"report written to {args.out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
