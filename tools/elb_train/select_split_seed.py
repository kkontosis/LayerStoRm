"""Held-out split seed selection (E4 addition, ratified 2026-08-02).

The INV-EPM-DATA sequence-level split is hash-bucketed per (seed, seq_key)
(dataset.sequence_split), so on small corpora the realized held-out BLOCK
fraction scatters around --held-out-fraction.  This tool re-derives the
split seed on an assembled shard directory by the standing PROCEDURE
(spec/MoE-SpeQ_NOTES.md §5b E4 / §5c): sweep seeds, keep seeds whose split
yields at least --min-held-sequences held-out sequences AND at least one
train sequence, and pick the one whose held-out BLOCK fraction is closest
to the target (tie-break: lowest seed).  The seed must be RE-DERIVED on
every new corpus — never reuse a previous corpus's seed or its literal
sequence list.

Usage:
  python3 tools/elb_train/select_split_seed.py --shards <shard-dir> \
      [--held-out-fraction 0.25] [--min-held-sequences 3] [--seeds 50]
Prints the sweep table and the chosen seed; exits 1 if no seed satisfies
the constraints (corpus too small — use run_baselines.py --in-sample).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # `python3 tools/elb_train/select_split_seed.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from elb_train import dataset
else:
    from . import dataset


def blocks_per_sequence(shard_dir: Path) -> dict[int, int]:
    """seq_key -> block count over all shards."""
    index = dataset.load_index(shard_dir)
    counts: dict[int, int] = {}
    for sh in index["shards"]:
        z = np.load(shard_dir / sh["path"])
        for sk in z["seq_key"]:
            sk = int(sk)
            counts[sk] = counts.get(sk, 0) + 1
    return counts


def sweep(shard_dir: Path, *, held_out_fraction: float,
          min_held_sequences: int, n_seeds: int) -> tuple[int | None, list]:
    counts = blocks_per_sequence(shard_dir)
    all_keys = sorted(counts)
    total_blocks = sum(counts.values())
    rows = []
    best = None  # (distance, seed)
    for seed in range(n_seeds):
        train, held = dataset.sequence_split(
            all_keys, held_out_fraction, seed=seed)
        held_blocks = sum(counts[k] for k in held)
        frac = held_blocks / total_blocks if total_blocks else 0.0
        ok = len(held) >= min_held_sequences and len(train) >= 1
        rows.append({"seed": seed, "train_seqs": len(train),
                     "held_seqs": len(held), "held_blocks": held_blocks,
                     "held_block_fraction": round(frac, 4), "eligible": ok})
        if ok:
            cand = (abs(frac - held_out_fraction), seed)
            if best is None or cand < best:
                best = cand
    return (best[1] if best else None), rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--held-out-fraction", type=float, default=0.25)
    ap.add_argument("--min-held-sequences", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=50,
                    help="sweep seeds 0..N-1")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write the sweep table + choice as JSON")
    args = ap.parse_args(argv)

    chosen, rows = sweep(args.shards,
                         held_out_fraction=args.held_out_fraction,
                         min_held_sequences=args.min_held_sequences,
                         n_seeds=args.seeds)
    print(f"{'seed':>4} {'train':>5} {'held':>4} {'held_blocks':>11} "
          f"{'held_frac':>9} eligible")
    for r in rows:
        mark = " <== chosen" if r["seed"] == chosen else ""
        print(f"{r['seed']:>4} {r['train_seqs']:>5} {r['held_seqs']:>4} "
              f"{r['held_blocks']:>11} {r['held_block_fraction']:>9.4f} "
              f"{str(r['eligible']):>8}{mark}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"chosen_seed": chosen,
                       "held_out_fraction": args.held_out_fraction,
                       "min_held_sequences": args.min_held_sequences,
                       "rows": rows}, f, indent=1)
    if chosen is None:
        print(f"NO eligible seed in 0..{args.seeds - 1} "
              f"(need >= {args.min_held_sequences} held-out sequences and "
              f">= 1 train sequence) — corpus too small; use "
              f"run_baselines.py --in-sample", file=sys.stderr)
        return 1
    print(f"chosen split_seed = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
