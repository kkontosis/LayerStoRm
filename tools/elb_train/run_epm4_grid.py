"""EPM-4 experiment campaign runner (PLAN Phase 29 EPM-4).

Runs a grid of EPM-3 training cells (each = tools/elb_train/train.py with
--set overrides off one base config), all on the SAME shards, SAME
sequence split, SAME seed and SAME step budget, then summarizes every
cell on the shared held-out split into a results table with the
deployment-relevant W-weighted aggregate (metrics.wmap_weighted_recall
against the DIRECTIVE W — the value map production will serve, regardless
of what W a cell trained with).

Grid spec JSON: {"cells": [{"name": ..., "overrides": ["k.path=json", ...],
"note": ...}, ...]}. Cells are resumable: a cell with an existing
<work>/<name>/eval_final.json is skipped (delete to re-run).

Run:
  python3 tools/elb_train/run_epm4_grid.py \
    --base tools/elb_train/configs/epm4-real-base.json \
    --grid tools/elb_train/configs/epm4-grid.json \
    --work build/epm4-grid --study studies/epm/epm4-grid \
    [--baselines <run_baselines out dir>] [--cells name1,name2]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402

from elb_train import dataset, metrics, wmap as wmap_mod  # noqa: E402


def deployment_wmap(shards: pathlib.Path) -> np.ndarray:
    """The DIRECTIVE default W at the corpus' (max_gamma, moe_layers) —
    the production value map every cell is scored against."""
    index = dataset.load_index(shards)
    return wmap_mod.build_wmap({"kind": "directive"}, index["max_gamma"],
                               index["moe_layers"])


def run_cell(base: pathlib.Path, work: pathlib.Path, cell: dict,
             extra_sets: list[str]) -> None:
    out = work / cell["name"]
    if (out / "eval_final.json").exists():
        print(f"[epm4-grid] {cell['name']}: eval_final.json exists — skip",
              flush=True)
        return
    cmd = [sys.executable, str(_HERE / "train.py"), "--config", str(base),
           "--out", str(out)]
    for s in list(cell.get("overrides", [])) + extra_sets:
        cmd += ["--set", s]
    print(f"[epm4-grid] {cell['name']}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))
    print(f"[epm4-grid] {cell['name']} done in "
          f"{time.monotonic() - t0:.0f}s", flush=True)


def summarize_cell(prefix: pathlib.Path, w_deploy: np.ndarray) -> dict:
    res = metrics.load_result_npz(prefix)
    row: dict = {}
    for m in (8, 16, 32):
        r, n = metrics.overall_recall(res, m)
        row[f"recall{m}"], row[f"n{m}"] = r, n
    row["w_score"], row["w_coverage"] = metrics.wmap_weighted_recall(
        res, w_deploy, 8)
    uc = res["union_coverage"]
    n = uc["n"]
    row["union_cov"] = (float(np.nansum(np.where(n > 0,
                                                 uc["mean"] * n, 0.0))
                              / n.sum()) if n.sum() else float("nan"))
    row["ece"] = res["calibration"]["ece"]
    row["per_position_recall8"] = metrics.per_position_recall(res, 8)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--grid", type=pathlib.Path, required=True)
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--study", type=pathlib.Path, required=True)
    ap.add_argument("--baselines", type=pathlib.Path, default=None,
                    help="run_baselines.py out dir (b0_prev/... prefixes) "
                         "to fold into the table as reference lines")
    ap.add_argument("--cells", default="",
                    help="comma list to run/summarize a subset")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY.PATH=JSON",
                    help="extra override applied to EVERY cell (e.g. a "
                         "campaign-wide step budget)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    with open(args.grid) as f:
        grid = json.load(f)
    cells = grid["cells"]
    if args.cells:
        keep = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in keep]

    args.work.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for cell in cells:
            run_cell(args.base, args.work, cell, args.overrides)

    # ── summarize ────────────────────────────────────────────────────────
    with open(args.base) as f:
        base_cfg = json.load(f)
    shards = pathlib.Path(base_cfg["data"]["shards"])
    w_deploy = deployment_wmap(shards)

    rows: dict[str, dict] = {}
    for cell in cells:
        prefix = args.work / cell["name"] / "eval_final"
        if not prefix.with_suffix(".json").exists():
            print(f"[epm4-grid] {cell['name']}: missing eval — skipped in "
                  f"summary", flush=True)
            continue
        rows[cell["name"]] = {"note": cell.get("note", ""),
                              **summarize_cell(prefix, w_deploy)}

    baseline_rows: dict[str, dict] = {}
    if args.baselines is not None:
        for name in ("b0_prev", "b0_anchor", "b1_best_tap", "b1_prior_tap",
                     "b2_linear_probe"):
            p = args.baselines / name
            if p.with_suffix(".npz").exists():
                baseline_rows[name] = summarize_cell(p, w_deploy)

    args.study.mkdir(parents=True, exist_ok=True)
    with open(args.study / "results.json", "w") as f:
        json.dump({"cells": rows, "baselines": baseline_rows,
                   "grid": str(args.grid), "base_config": str(args.base),
                   "work_dir": str(args.work)}, f, indent=2)

    # Small artifacts into the study dir (npz/json per cell).
    for name in rows:
        for ext in (".json", ".npz"):
            src = args.work / name / ("eval_final" + ext)
            if src.exists():
                shutil.copy2(src, args.study / f"cell_{name}{ext}")

    # Markdown table.
    hdr = ("| cell | recall@8 | W-score | recall@16 | recall@32 | "
           "union | ECE | note |")
    sep = "|---" * 8 + "|"
    lines = [hdr, sep]

    def fmt(row: dict, name: str, note: str) -> str:
        return (f"| {name} | {row['recall8']:.4f} | {row['w_score']:.4f} "
                f"| {row['recall16']:.4f} | {row['recall32']:.4f} "
                f"| {row['union_cov']:.4f} | {row['ece']:.4f} | {note} |")

    for name in sorted(rows, key=lambda n: -rows[n]["w_score"]):
        lines.append(fmt(rows[name], name, rows[name]["note"]))
    for name, row in baseline_rows.items():
        lines.append(fmt(row, f"*{name}*", "reference"))
    table = "\n".join(lines)
    with open(args.study / "results_table.md", "w") as f:
        f.write(table + "\n")
    print(table, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
