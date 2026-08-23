#!/usr/bin/env python3
"""P-26 unified I8 deployment fit driver — constants + M2v2 params + policy
weights from ONE invocation, each parameter class by its mathematically-correct
method (spec/tickets/P-26_I8_DEPLOYMENT_FIT.md):

  constants  LoaderConstants correction: supervised regression against measured
             per-layer walls — delegates to tools/loader_xray/trainer_apply.py
             (needs --dump + --trace + --calib).
  m2         M2v2 exposed-wall params (s,o0,oc,hsat,cpw,g_c,b0): supervised
             regression — delegates to tools/loader_xray/exposed_model.py
             --form v2 (needs --dump + --trace + --calib). Config-scoped:
             refit per deployment, never reuse across configs.
  policy     freq/reuse policy weights (freq_w, freq_decay, reuse_w,
             reuse_tau): closed-loop BLACK-BOX optimization, never regression
             (INV-LOADER-OBJECTIVE-MYOPIC) — a coarse grid replayed CPU-only
             through the offline sim (loader_offline_sim policy_lab/
             trajectory_sim machinery), ranked by sim hit rate. The sim is
             open-loop (cannot see trajectory forks): the KEEPER stays the
             final gate — this driver prints the single-run keeper confirm
             recipe and NEVER launches a keeper or touches a GPU.

Artifacts land in --out-dir:
  loader_constants_trained.json   (existing trainer_apply schema)
  exposed_params_v2.json          (existing exposed_model schema)
  policy_params.json              (schema = the frozen epoch-2 fixture
      loader_offline_sim/fixtures/2026-07-18_epoch2/policy_params_reef_4gpu.json,
      plus a provenance block: epoch tag, input capture paths, winner sim hit,
      keeper_fingerprint=null until keeper-confirmed).

Ground rules enforced here:
  * refuses to overwrite existing outputs without --force;
  * refuses to write anywhere under loader_offline_sim/fixtures/ (FROZEN
    epochs — a refit writes NEW artifacts, never edits fixtures);
  * CPU-only: no keeper launch, no GPU, ever.

Example (policy only, on the committed sim assets):
  python3 tools/loader_fit/deploy_fit.py --only policy \
      --dump tests/integration/loader_offline_sim/assets/traj_4gpu_canonical.jsonl.gz \
      --calib tests/assets/gpu_loader_calibration_4gpu_hbm.json \
      --out-dir /tmp/fit_out
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM_DIR = REPO / "tests" / "integration" / "loader_offline_sim"
XRAY_DIR = REPO / "tools" / "loader_xray"
FIXTURES_DIR = (SIM_DIR / "fixtures").resolve()

sys.path.insert(0, str(SIM_DIR))
import policy_lab                     # noqa: E402  (offline sim machinery)
from trajectory_sim import load_traj  # noqa: E402

# Coarse default grid, centered on the landed plateau optimum 60/2000/300
# (DETHRONE_AFFINITY_IDEAS.md UPDATE 9 — same corner shape as the keeper
# regrid: freq inverted-U, reuse/tau corners). freq_decay stays fixed (0.1)
# unless --freq-decay: the keeper sweeps never separated it from freq_w.
GRID_FREQ_W = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0]
GRID_REUSE_W = [1000.0, 2000.0, 4000.0]
GRID_REUSE_TAU = [150.0, 300.0, 600.0]


def fail(msg: str) -> None:
    sys.exit(f"deploy_fit: {msg}")


def guard_out(path: Path, force: bool) -> Path:
    """Refuse frozen-fixture writes always; refuse overwrite without --force."""
    rp = path.resolve()
    if rp == FIXTURES_DIR or FIXTURES_DIR in rp.parents:
        fail(f"refusing to write into frozen fixtures ({rp}); "
             "fixtures are FROZEN epochs — a refit writes NEW artifacts")
    if rp.exists() and not force:
        fail(f"output exists: {rp} — pass --force to overwrite")
    return rp


def run_stage(name: str, cmd: list[str]) -> None:
    print(f"\n── {name}: {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        fail(f"{name} stage failed (exit {r.returncode})")


# ── policy grid (black-box, offline sim) ─────────────────────────────────────

def policy_grid(args, out_path: Path) -> None:
    caps = [int(x) for x in args.caps.split(",")]
    if len(caps) != 4:
        fail("the offline sim machinery is 4-GPU (policy_lab placers); "
             f"--caps must have 4 entries, got {len(caps)}")
    policy_lab.CAPS = caps  # module-global consumed by the placers
    calib = json.load(open(args.calib))
    if calib.get("num_devices", len(caps)) != len(caps):
        fail(f"calib num_devices={calib.get('num_devices')} != caps {len(caps)}")
    recs = policy_lab.flatten(load_traj(args.dump))
    print(f"policy grid: trace {args.dump} ({len(recs)} records), "
          f"calib {args.calib}, caps {caps}")
    fd = args.freq_decay
    grid = list(itertools.product(args.grid_freq_w, args.grid_reuse_w,
                                  args.grid_reuse_tau))
    print(f"  {len(grid)} candidates (freq_w x reuse_w x reuse_tau, "
          f"freq_decay fixed {fd}) — engine-faithful arm: freq-protected "
          "eviction + reuse place cost\n")
    results = []
    t0 = time.time()
    for i, (fw, rw, tau) in enumerate(grid):
        # Engine analogue: eviction score = recency + freq_w * decayed-freq
        # (make_sc_freq == the dispatcher's w*f recency bonus; fw=0 -> LRU),
        # placement = transfer makespan + reuse victim cost w/(1+age/tau)
        # (make_place_reuse == the landed place_cons term; rw=0 -> makespan).
        scorer = policy_lab.make_sc_freq(decay=fd, w=fw)
        placer = policy_lab.make_place_reuse(w=rw, tau=tau, calib=calib)
        r = policy_lab.replay(recs, calib, scorer, placer, caps=caps)
        results.append({"freq_w": fw, "freq_decay": fd, "reuse_w": rw,
                        "reuse_tau": tau, **r})
        print(f"  [{i + 1:2d}/{len(grid)}] fw={fw:<6g} rw={rw:<6g} "
              f"tau={tau:<5g} -> hit={r['hit']:.4f} "
              f"fetch/tok={r['fetch_tok']:.1f} pred={r['pred_ms']:.1f} ms/tok",
              flush=True)
    print(f"\n  grid done in {time.time() - t0:.1f} s")

    # Rank: sim hit rate desc (the keeper-transferable signal — wall tracks hit
    # at ~0.15 tok/s per pp, UPDATE 9); tie-break predicted fetch-ms asc.
    results.sort(key=lambda r: (-r["hit"], r["pred_ms"]))
    print("\n── ranked candidates (sim hit desc; SIM RANKS, KEEPER DECIDES) ──")
    print(f"  {'rank':>4} {'freq_w':>7} {'freq_d':>7} {'reuse_w':>8} "
          f"{'tau':>6} {'hit':>7} {'fetch/tok':>9} {'pred ms/tok':>11}")
    for rank, r in enumerate(results, 1):
        mark = "  <- winner" if rank == 1 else ""
        print(f"  {rank:>4} {r['freq_w']:>7g} {r['freq_decay']:>7g} "
              f"{r['reuse_w']:>8g} {r['reuse_tau']:>6g} {r['hit']:>7.4f} "
              f"{r['fetch_tok']:>9.1f} {r['pred_ms']:>11.1f}{mark}")

    win = results[0]
    artifact = {
        "_comment": "P-26 deploy_fit policy weights — sim-ranked winner; "
                    "keeper confirm PENDING (keeper_fingerprint stays null "
                    "until the single-run keeper gate confirms).",
        "epoch": args.epoch,
        "freq_w": win["freq_w"],
        "freq_decay": win["freq_decay"],
        "reuse_w": win["reuse_w"],
        "reuse_tau": win["reuse_tau"],
        "provenance": {
            "epoch": args.epoch,
            "generated_by": "tools/loader_fit/deploy_fit.py",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "inputs": {"dump": str(Path(args.dump).resolve()),
                       "calib": str(Path(args.calib).resolve()),
                       "caps": caps},
            "grid": {"freq_w": args.grid_freq_w,
                     "reuse_w": args.grid_reuse_w,
                     "reuse_tau": args.grid_reuse_tau,
                     "freq_decay": fd},
            "sim_hit": round(win["hit"], 4),
            "sim_fetch_per_tok": round(win["fetch_tok"], 2),
            "sim_pred_ms_per_tok": round(win["pred_ms"], 2),
            "keeper_fingerprint": None,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {out_path}")
    print(json.dumps(artifact, indent=2))

    print("\n── KEEPER CONFIRM RECIPE (run ONE keeper, by hand — this driver "
          "never launches keepers/GPUs) ──")
    print("  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \\")
    print("    KEEPER52_EP=4 KEEPER52_REEF_ORCH=1 \\")
    print("    LS_LOADER_CALIB=gpu_loader_calibration_4gpu_hbm.json \\")
    print(f"    LS_LOADER_POLICY={out_path} \\")
    print("    ./build/tests/integration/keeper52_test \\")
    print("      --gtest_filter='Keeper52Test."
          "HundredTokenDecodeFetchAndRun_FullFit_EP2_GLM52'")
    print("  Judge gate-on by keeper fingerprint+wall ONLY (one keeper at a "
          "time; interleave a control for wall claims); then backfill")
    print("  provenance.keeper_fingerprint in the artifact with the confirmed "
          "hit fingerprint.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="P-26 unified I8 deployment fit (constants + M2v2 + "
                    "policy weights). CPU-only; never launches keepers/GPUs.")
    ap.add_argument("--only", action="append",
                    choices=["constants", "m2", "policy"], default=None,
                    help="run only this stage (repeatable; default: all)")
    ap.add_argument("--dump", required=True,
                    help="LS_LOADER_SHADOW_DUMP JSONL[.gz] capture (all stages)")
    ap.add_argument("--calib", required=True,
                    help="baseline LoaderConstants calibration JSON (all stages)")
    ap.add_argument("--trace", default=None,
                    help="LS_PERF_TRACE_OUT CSV (required for constants + m2)")
    ap.add_argument("--mode", default="act", choices=["act", "aff"],
                    help="the run mode the dump came from (m2 join; default act)")
    ap.add_argument("--val", action="append", default=[],
                    help="m2: extra <jsonl>:<csv>:<act|aff> validation specs")
    ap.add_argument("--caps", default="203,203,443,443",
                    help="per-GPU stable-zone slot caps (policy sim)")
    ap.add_argument("--freq-decay", type=float, default=0.1,
                    help="fixed freq_decay for the policy grid (default 0.1)")
    ap.add_argument("--grid-freq-w", type=lambda s: [float(x) for x in s.split(",")],
                    default=GRID_FREQ_W, help="comma list (policy grid)")
    ap.add_argument("--grid-reuse-w", type=lambda s: [float(x) for x in s.split(",")],
                    default=GRID_REUSE_W, help="comma list (policy grid)")
    ap.add_argument("--grid-reuse-tau", type=lambda s: [float(x) for x in s.split(",")],
                    default=GRID_REUSE_TAU, help="comma list (policy grid)")
    ap.add_argument("--epoch", default=None,
                    help="epoch tag stamped into the artifacts "
                    "(default: <today>-deployfit)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for the artifact bundle")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output artifacts")
    args = ap.parse_args()

    stages = args.only or ["constants", "m2", "policy"]
    if args.epoch is None:
        args.epoch = f"{datetime.date.today().isoformat()}-deployfit"
    out_dir = Path(args.out_dir).resolve()
    if out_dir == FIXTURES_DIR or FIXTURES_DIR in out_dir.parents:
        fail(f"refusing --out-dir inside frozen fixtures ({out_dir})")
    out_dir.mkdir(parents=True, exist_ok=True)
    if ("constants" in stages or "m2" in stages) and not args.trace:
        fail("--trace (perf_trace CSV) is required for the constants and m2 "
             "stages; use --only policy for a trace-free policy-only fit")

    print(f"deploy_fit: epoch={args.epoch} stages={stages} out={out_dir}")

    if "constants" in stages:
        out = guard_out(out_dir / "loader_constants_trained.json", args.force)
        run_stage("constants (trainer_apply regression)", [
            str(XRAY_DIR / "trainer_apply.py"),
            "--in-calib", args.calib, "--model-jsonl", args.dump,
            "--trace", args.trace, "--out-calib", str(out)])
        print(f"wrote {out}")

    if "m2" in stages:
        out = guard_out(out_dir / "exposed_params_v2.json", args.force)
        cmd = [str(XRAY_DIR / "exposed_model.py"), "--form", "v2",
               "--calib", args.calib,
               "--train", f"{args.dump}:{args.trace}:{args.mode}",
               "--out", str(out)]
        for v in args.val:
            cmd += ["--val", v]
        run_stage("m2 (exposed_model --form v2 regression)", cmd)
        print(f"wrote {out}  (config-scoped — refit per deployment)")

    if "policy" in stages:
        out = guard_out(out_dir / "policy_params.json", args.force)
        policy_grid(args, out)


if __name__ == "__main__":
    main()
