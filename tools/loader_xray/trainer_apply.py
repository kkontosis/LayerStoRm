"""CLI: join -> fit -> apply -> write a corrected LoaderConstants JSON (I8 trainer
feedback loop).

Reads the baseline LoaderConstants calibration JSON, the shadow model JSONL dump
(LS_LOADER_SHADOW_DUMP) and the perf_trace CSV (LS_PERF_TRACE_OUT), joins them into
the per-(cmd_seq,layer) dataset, fits the pluggable cost model's correction
coefficients (trainer.train), bakes those corrections back into the constants
(trainer.apply_params_to_constants), and writes a NEW LoaderConstants JSON in the
SAME on-disk schema (loadable by the engine via gpu_loader.calibration_path).

The written JSON carries the new global ``fixed_overhead_us`` field (the additive
offsets the per-rate model structurally misses) plus the scaled per-device transfer
rates / compute curve / bank egress / recon overheads. Re-running the engine with
calibration_path=<out> makes the loader solver predict as if trained.

Usage:
    python3 trainer_apply.py \
        --in-calib  gpu_loader_calibration_5090x2.json \
        --model-jsonl loader_shadow.jsonl \
        --trace     perf_trace.csv \
        --model     current \
        --out-calib trained.json \
        [--iters 300] [--lr 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys

from joiner import join
from model import make_model
from trainer import train, apply_params_to_constants, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit + bake loader corrections into LoaderConstants JSON.")
    ap.add_argument("--in-calib", required=True, help="baseline LoaderConstants JSON")
    ap.add_argument("--model-jsonl", required=True, help="LS_LOADER_SHADOW_DUMP JSONL")
    ap.add_argument("--trace", required=True, help="LS_PERF_TRACE_OUT CSV")
    ap.add_argument("--model", default="current", choices=["current", "contention_bank"])
    ap.add_argument("--out-calib", required=True, help="output corrected LoaderConstants JSON")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args(argv)

    with open(args.in_calib) as fh:
        calib = json.load(fh)

    rows, stats = join(args.model_jsonl, args.trace)
    print(json.dumps({"join": stats}, indent=2), file=sys.stderr)
    if not rows:
        print("trainer_apply: no joined rows — nothing to fit", file=sys.stderr)
        return 2

    M = max(int(r["M"]) for r in rows)
    mdl = make_model(args.model, M)
    res = train(mdl, rows, iters=args.iters, lr=args.lr)
    print(why(res, mdl.init_params()), file=sys.stderr)

    out = apply_params_to_constants(mdl, res.params, calib)
    with open(args.out_calib, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        json.dumps({
            "wrote": args.out_calib,
            "model": res.model,
            "loss0": res.loss0,
            "loss": res.loss,
            "fixed_overhead_us": out.get("fixed_overhead_us", 0.0),
        }, indent=2),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
