"""EPM-5 corpus batch driver (Phase B, studies/epm/epm5-corpus).

Runs the champion-recipe collector (collect_corpus.py --recipe champion)
over the prompts_epm5.json set in RESUMABLE batches of --batch-size
prompts (fresh engine boot per batch — the per-request drift finding),
then incrementally append-assembles each batch into ONE shard dir
(dataset.assemble_shards run_idx_offset=batch append=True
store_logits=False compress=True) and deletes the batch's raw .bin
files (manifest.jsonl / provenance.json / collection_results.json are
kept — health provenance). Disk plan: peak ~= shards_total + one batch
of raw (PROGRESS.md Phase A).

State lives in <corpus-root>/progress.json — a batch is "collected"
once the collector exits 0 and "assembled" once shards merged + raw
deleted; re-running the driver skips completed batches, re-assembles a
collected-but-unassembled batch, and re-runs a half-collected batch
from scratch (its dump dir is wiped — the engine run is atomic per
dump dir).

Discipline per run: GPUs must be process-free before boot
(nvidia-smi compute-apps poll), --drain-s cool-down after every engine
exit (M6 ops finding), engine log kept per batch under
<corpus-root>/logs/.

Health gates (every --health-every batches + at stop): per-position
label coverage from dataset_stats on the merged shards, acceptance
ratio band from the batch manifests, n-gram loop detection over
gen_token_ids (degenerate greedy loops inflate acceptance and poison
the label distribution). Violations STOP the driver (exit 3) — human
judgment, not auto-pruning.

Run (repo root):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 tools/elb_train/collect_epm5.py \
      --corpus-root build/epm5-corpus --shards build/epm5-shards \
      --prompts tools/elb_train/prompts_epm5.json
Stop conditions: --target-blocks reached (default 105000), prompt set
exhausted, health-gate violation, or collector failure.
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
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

from elb_train import dataset  # noqa: E402


def gpus_free() -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return out == ""


def wait_gpus_free(timeout_s: float = 600.0) -> None:
    t0 = time.monotonic()
    while not gpus_free():
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError("GPUs still busy after "
                               f"{timeout_s:.0f}s — refusing to boot "
                               "(serial GPU discipline)")
        time.sleep(5)


def detect_loops(token_ids: list[int], max_n: int = 8,
                 window: int = 120, threshold: float = 0.9) -> bool:
    """True when the tail `window` tokens are >= threshold covered by a
    repeating cycle of length <= max_n (greedy degeneration signature)."""
    tail = token_ids[-window:]
    if len(tail) < 2 * max_n:
        return False
    for n in range(1, max_n + 1):
        rep = sum(1 for i in range(n, len(tail)) if tail[i] == tail[i - n])
        if rep / (len(tail) - n) >= threshold:
            return True
    return False


def loop_onset(token_ids: list[int], max_n: int = 8,
               threshold: float = 0.9, min_len: int = 40) -> int | None:
    """Earliest generated-token index from which the suffix is >=threshold
    covered by a cycle of length <= max_n — the onset of a degenerate
    greedy-repetition tail (forced-length generation past the natural end
    of a bounded-answer prompt). None = no degenerate tail.

    Coarse scan (step 8) then refine — the onset only needs block-level
    precision (blocks are ~2-5 tokens). Requires the looping suffix to be
    at least `min_len` tokens (a short accidental repeat is not a loop)."""
    n = len(token_ids)

    def _looped(start: int) -> bool:
        seg = token_ids[start:]
        if len(seg) < max(min_len, 2 * max_n):
            return False
        for p in range(1, max_n + 1):
            rep = sum(1 for i in range(p, len(seg)) if seg[i] == seg[i - p])
            if rep / (len(seg) - p) >= threshold:
                return True
        return False

    coarse = None
    for start in range(0, n - min_len, 8):
        if _looped(start):
            coarse = start
            break
    if coarse is None:
        return None
    for start in range(max(0, coarse - 8), coarse + 1):
        if _looped(start):
            return start
    return coarse


def batch_health(run_dir: pathlib.Path) -> dict:
    """Per-batch manifest/results health numbers."""
    h: dict = {"blocks": 0, "attempted": 0, "accepted": 0, "loops": []}
    mpath = run_dir / "manifest.jsonl"
    if mpath.is_file():
        for line in open(mpath):
            e = json.loads(line)
            h["blocks"] += 1
            h["attempted"] += int(e.get("attempted_len", 0))
            h["accepted"] += int(e.get("accepted_len", 0))
    # Loop onsets per sequence (submission order = engine seq_id, 1-based).
    # prov gives each prompt's token count so a generated-token onset maps
    # to an absolute cutoff anchor_pos = prompt_len + onset.
    prov = {}
    ppath = run_dir / "provenance.json"
    if ppath.is_file():
        prov = json.load(open(ppath))
    prompts = prov.get("prompts", [])
    h["truncate"] = {}          # seq_id -> cutoff anchor_pos (drop >=)
    rpath = run_dir / "collection_results.json"
    if rpath.is_file():
        for sid, r in enumerate(json.load(open(rpath)), start=1):
            ids = r.get("gen_token_ids") or []
            on = loop_onset(ids)
            if on is None:
                continue
            h["loops"].append(r["name"])
            frac = on / max(1, len(ids))
            plen = (len(prompts[sid - 1]["token_ids"])
                    if sid - 1 < len(prompts) else 0)
            h["truncate"][str(sid)] = {
                "name": r["name"], "onset": on, "good_fraction": frac,
                "cutoff_anchor": plen + on}
    h["acceptance_ratio"] = (h["accepted"] / h["attempted"]
                             if h["attempted"] else 0.0)
    return h


def load_progress(path: pathlib.Path) -> dict:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {"batches": {}}


def save_progress(path: pathlib.Path, prog: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=1)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", default="build/epm5-corpus")
    ap.add_argument("--shards", default="build/epm5-shards")
    ap.add_argument("--prompts", default="tools/elb_train/prompts_epm5.json")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--target-blocks", type=int, default=105000)
    ap.add_argument("--drain-s", type=float, default=150.0)
    ap.add_argument("--health-every", type=int, default=10)
    ap.add_argument("--acceptance-band", default="0.10,0.60",
                    help="min,max sane aggregate accepted/attempted")
    ap.add_argument("--min-good", type=float, default=0.25,
                    help="a looped sequence with coherent-prefix fraction "
                         "below this is treated as a broken prompt "
                         "(hard-stop for review); at/above it the "
                         "degenerate tail is truncated and kept")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after N batches this invocation (0 = all)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="do not delete raw .bin files after assembly")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.corpus_root).resolve()
    shards = pathlib.Path(args.shards).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    acc_lo, acc_hi = (float(x) for x in args.acceptance_band.split(","))

    with open(args.prompts) as f:
        prompts = json.load(f)
    batches = [prompts[i:i + args.batch_size]
               for i in range(0, len(prompts), args.batch_size)]
    prog_path = root / "progress.json"
    prog = load_progress(prog_path)

    def total_blocks() -> int:
        idx = shards / "shards_index.json"
        if idx.is_file():
            with open(idx) as f:
                return json.load(f)["num_blocks"]
        return 0

    done_this_invocation = 0
    for bi, batch in enumerate(batches):
        key = f"batch_{bi:03d}"
        st = prog["batches"].get(key, {})
        if st.get("assembled"):
            continue
        if total_blocks() >= args.target_blocks:
            print(f"[epm5-drive] target reached: {total_blocks()} blocks "
                  f">= {args.target_blocks}", flush=True)
            break
        if args.max_batches and done_this_invocation >= args.max_batches:
            print(f"[epm5-drive] max-batches {args.max_batches} reached "
                  f"this invocation", flush=True)
            break

        run_dir = root / f"run_{bi:03d}"
        if not st.get("collected"):
            # (Re-)collect: wipe any half-collected dump.
            if run_dir.exists():
                shutil.rmtree(run_dir)
            bp = logs / f"{key}_prompts.json"
            with open(bp, "w") as f:
                json.dump(batch, f, indent=1)
            wait_gpus_free()
            print(f"[epm5-drive] {key}: collecting "
                  f"{[p['name'] for p in batch]}", flush=True)
            t0 = time.monotonic()
            log_f = open(logs / f"{key}.log", "w")
            r = subprocess.run(
                [sys.executable, str(_HERE / "collect_corpus.py"),
                 "--dump-dir", str(run_dir), "--prompts", str(bp),
                 "--recipe", "champion"],
                stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT))
            log_f.close()
            dt = time.monotonic() - t0
            print(f"[epm5-drive] {key}: collector exit {r.returncode} "
                  f"in {dt:.0f}s — draining {args.drain_s:.0f}s",
                  flush=True)
            time.sleep(args.drain_s)
            if r.returncode != 0:
                print(f"[epm5-drive] {key}: COLLECTOR FAILED — stopping "
                      f"(log: {logs / (key + '.log')})", flush=True)
                return 2
            h = batch_health(run_dir)
            st.update(collected=True, wall_s=round(dt, 1), health=h)
            prog["batches"][key] = st
            save_progress(prog_path, prog)
            if not (acc_lo <= h["acceptance_ratio"] <= acc_hi):
                print(f"[epm5-drive] {key}: acceptance "
                      f"{h['acceptance_ratio']:.3f} outside "
                      f"[{acc_lo},{acc_hi}] — stopping for review",
                      flush=True)
                return 3

        # Recompute health if the stored record predates the truncate
        # field (a batch collected by an older driver) — the run_dir still
        # carries collection_results/provenance/manifest.
        health = st.get("health") or {}
        if "truncate" not in health and run_dir.is_dir():
            health = batch_health(run_dir)
            st["health"] = health
            prog["batches"][key] = st
            save_progress(prog_path, prog)

        # Resolve the per-sequence block filter from degenerate-loop
        # findings. Two regimes (studies/epm/epm5-corpus PROGRESS.md):
        #  - PARTIAL loop (good fraction >= --min-good): the model wrote a
        #    coherent prefix then fell into a repetition tail (forced-length
        #    generation past the natural end). Keep the prefix, TRUNCATE at
        #    cutoff_anchor = prompt_len + onset.
        #  - MOSTLY loop (good fraction < --min-good): the whole sequence is
        #    degenerate (e.g. GLM completion-mode bilingual drift into
        #    hashtag spam — even the "prefix" is garbage). DROP it entirely
        #    (cutoff 0) and CONTINUE — one bad prompt among hundreds must
        #    not halt the campaign. Recorded in progress["dropped_seqs"] for
        #    review. SAFETY: if a MAJORITY of a batch is mostly-loop,
        #    something is systemically wrong — hard-stop.
        trunc = health.get("truncate", {})
        cutoffs: dict[int, int] = {}
        dropped: list[str] = []
        truncated: list[tuple] = []
        for sid, t in trunc.items():
            if t["good_fraction"] < args.min_good:
                cutoffs[int(sid)] = 0            # drop all blocks
                dropped.append(t["name"])
            else:
                cutoffs[int(sid)] = t["cutoff_anchor"]
                truncated.append((t["name"], round(t["good_fraction"], 2)))
        if len(dropped) * 2 >= max(1, len(batch)):
            print(f"[epm5-drive] {key}: SYSTEMIC — {len(dropped)}/{len(batch)} "
                  f"sequences mostly-loop {dropped}; stopping for review",
                  flush=True)
            return 3
        if dropped:
            print(f"[epm5-drive] {key}: DROPPING degenerate sequences "
                  f"{dropped} (good fraction < {args.min_good}) — continuing",
                  flush=True)
            prog.setdefault("dropped_seqs", []).extend(
                f"{key}:{n}" for n in dropped)
        if truncated:
            print(f"[epm5-drive] {key}: truncating degenerate tails "
                  f"(kept fraction) {truncated}", flush=True)
        block_filter = (
            (lambda seq_id, anchor_pos:
             anchor_pos < cutoffs.get(int(seq_id), 1 << 62))
            if cutoffs else None)

        # Assemble-append + raw cleanup (idempotent per batch).
        print(f"[epm5-drive] {key}: assembling into {shards}", flush=True)
        idx = dataset.assemble_shards(
            run_dir, shards, run_idx_offset=bi,
            append=(shards / "shards_index.json").is_file(),
            store_logits=False, compress=True, block_filter=block_filter)
        if not args.keep_raw:
            # Delete ONLY the feature dump (921 KB/block — the disk hog;
            # its bytes live on in the shards). epm_routing.bin
            # (~130 KB/block) is KEPT: the B0 temporal-locality baselines
            # — the Phase-C bar AND the b0_prev-hybrid arm's candidate
            # source — read it directly (baselines.py: records exist for
            # EVERY decoded position, incl. positions the shards never
            # label).
            p = run_dir / "epm_blocks.bin"
            if p.is_file():
                p.unlink()
        st.update(assembled=True, corpus_blocks=idx["num_blocks"],
                  corpus_sequences=idx["num_sequences"])
        prog["batches"][key] = st
        save_progress(prog_path, prog)
        done_this_invocation += 1
        print(f"[epm5-drive] {key}: corpus now {idx['num_blocks']} blocks "
              f"/ {idx['num_sequences']} seqs", flush=True)

        if done_this_invocation % args.health_every == 0:
            stats = dataset.dataset_stats(shards)
            cov = stats.get("position_label_coverage", [])
            print(f"[epm5-drive] health: blocks={stats.get('num_blocks')} "
                  f"cov[k0]={cov[0] if cov else None} "
                  f"cov[k-1]={cov[-1] if cov else None}", flush=True)
            if cov and cov[0] < 0.9:
                print("[epm5-drive] k=0 label coverage < 0.9 — stopping "
                      "for review", flush=True)
                return 3

    print(f"[epm5-drive] done: {total_blocks()} blocks in corpus",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
