"""P11 Stage-0 END-TO-END pipeline: model-agnostic, step-based, resumable
(PLAN.md Phase 30 P11; spec/reports/EXPOSED_BYTE_CALCULUS.md §3-P11(6)).

One JSON config describes a model + corpus; the runner executes ordered,
idempotent steps with a persistent state file and re-entry:

  collect      engine corpus collection (wraps collect_epm5.py — the
               EPM-5 recipe: champion env, sequential requests, loop
               truncate/drop, per-batch append-assembly, health gates).
               GPU + engine required; disabled by default and REFUSES to
               run when the target corpus already exists (--force to
               override). The pipeline records git rev + config snapshot
               (closing the provenance gap in the underlying scripts).
  assemble     raw run units -> shards (only when collect was run
               outside the batch driver; collect_epm5 assembles inline).
  geometry     load shards_index, validate routing streams (memmap open
               + layer-set match), record J/E/K/G.
  sidecars     prev_top_ids_%05d.npy + recur_%05d.npz if missing.
  split        sequence split seed: pinned in config, or re-derived by
               the select_split_seed procedure (NEVER reuse another
               corpus's seed — INV-EPM-DATA / MoE-SpeQ_NOTES §5b).
  bars         b0_prev model-side bars (compute_b0_bars) if missing.
  router_bank  frozen router tensors from the model checkpoint
               (name templates in config — DeepSeek/GLM default).
  verify       stage0.py --do verify: join/bar/rank gates. Hard stop.
  evaluate     stage0.py --do eval: decomposition + bank + manifest +
               evict dumps.
  evict        stage0.py --do evict (process-parallel sims).
  verdict      merge step outputs, evaluate the §3-P11(6) kill gate
               ((i) within-pool headroom), rank the bank, write
               verdict.json + refresh the study dir.

Usage:
  python3 tools/elb_train/stage0_pipeline.py \
      --config tools/elb_train/configs/stage0-glm52.json \
      [--from STEP] [--to STEP] [--only STEP] [--force] [--list]

Steps already 'done' in <workdir>/pipeline_state.json are skipped unless
forced or explicitly re-entered via --from/--only. Every step records
started/ended timestamps and a small info dict; failures mark the step
'failed' and stop the run (resume re-executes the failed step).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from elb_train import dataset, glm_router, recurrence  # noqa: E402

STEP_ORDER = ("collect", "assemble", "geometry", "sidecars", "split",
              "bars", "router_bank", "verify", "evaluate", "evict",
              "verdict", "stageb_dump", "stageb_fit", "stageb_eval",
              "stageb_evict", "stageb_verdict",
              "stagec_run", "stagec_evict", "stagec_verdict",
              "stagecprime_prep", "stagecprime_run",
              "stagecprime_verdict", "stagecprime_cppfix")


# ── config / state ───────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "name": "stage0",
    "corpus": {"shards": None, "routing": None, "dump_root": None},
    "collection": {"enabled": False},
    "router": {"checkpoint": None,
               "weight_tmpl": "model.layers.{j}.mlp.gate.weight",
               "bias_tmpl":
               "model.layers.{j}.mlp.gate.e_score_correction_bias"},
    "split": {"seed": None, "held_fraction": 0.25, "select_if_unset": True,
              "min_held_sequences": 3, "candidate_seeds": 50},
    "stage0": {"windows": "prevpos,prevchunk,trail16,trail64",
               "sigma_c": "0,0.5,1,2", "pool_m": "8,16,32,64",
               "deep_lo": None, "deep_hi": None, "sigma_seqs": 50,
               "evict_max_seqs": 24, "evict_caps": "700,1052",
               "headline_window": "trail16", "headline_c": 1.0},
    "gates": {"within_headroom_min": 0.035,
              "rank_set_agreement_min": 0.99, "bar_tolerance": 1e-3},
    "stageb": {"enabled": True, "stride": 32, "fit_cap": 8_000_000,
               "gbm_rounds": 400, "evict_max_seqs": 24,
               "evict_caps": "700,1052", "jobs": 24},
    "stagec": {"enabled": False, "jobs": 24, "device": "",
               "evict_max_seqs": 24, "evict_caps": "700,1052"},
    "stagecprime": {"enabled": False, "device": "", "exps": "all",
                    "memo_buckets": "4096",
                    "embed_tensor": "model.embed_tokens.weight",
                    "champion_exp": "x18ship",
                    "export_configs": "x18ship,x17ship",
                    "publish_model_dir": "",
                    "publish_model_dirs": "",
                    "cpp_fixture_out":
                    "tests/assets/expert_ridge_parity.safetensors"},
    "workdir": None,
    "study_dir": None,
}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    cfg = copy.deepcopy(_DEFAULTS)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    for req in ("workdir",):
        if not cfg.get(req):
            raise ValueError(f"config missing required field {req!r}")
    if not cfg["corpus"]["shards"]:
        raise ValueError("config missing corpus.shards")
    return cfg


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class State:
    def __init__(self, workdir: Path):
        self.path = workdir / "pipeline_state.json"
        self.data = {"steps": {}, "created": _now()}
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        tmp.replace(self.path)

    def step(self, name: str) -> dict:
        return self.data["steps"].get(name, {})

    def mark(self, name: str, status: str, info: dict | None = None) -> None:
        rec = self.data["steps"].setdefault(name, {})
        rec["status"] = status
        rec.setdefault("started", _now())
        if status in ("done", "failed", "skipped"):
            rec["ended"] = _now()
        if info is not None:
            rec["info"] = info
        self.save()


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_HERE, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _run_cli(argv: list[str], log: Path) -> None:
    """Run a child python tool, teeing output to a log file; raise on
    failure with the log tail inlined."""
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable] + argv, stdout=f,
                           stderr=subprocess.STDOUT,
                           cwd=_HERE.parent.parent)
    if p.returncode != 0:
        tail = "".join(open(log, encoding="utf-8").readlines()[-15:])
        raise RuntimeError(f"{argv[1] if len(argv) > 1 else argv[0]} "
                           f"exited {p.returncode}; log tail:\n{tail}")


# ── steps ────────────────────────────────────────────────────────────────────

def step_collect(cfg: dict, st: State, work: Path, force: bool) -> dict:
    col = cfg["collection"]
    if not col.get("enabled"):
        return {"skipped": "collection.enabled is false — using the "
                           "existing corpus"}
    shards = Path(cfg["corpus"]["shards"])
    if (shards / "shards_index.json").is_file() and not force:
        raise RuntimeError(
            f"collection.enabled but {shards} already holds an assembled "
            f"corpus — refusing to re-collect (--force to override)")
    import os
    env = dict(os.environ)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if col.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = col["cuda_visible_devices"]
    driver = col.get("driver", "tools/elb_train/collect_epm5.py")
    argv = [driver,
            "--corpus-root", col.get("corpus_root", "build/epm5-corpus"),
            "--shards", str(shards),
            "--prompts", col.get("prompts",
                                 "tools/elb_train/prompts_epm5.json")]
    for flag, key in (("--batch-size", "batch_size"),
                      ("--target-blocks", "target_blocks"),
                      ("--drain-s", "drain_s"),
                      ("--acceptance-band", "acceptance_band"),
                      ("--min-good", "min_good"),
                      ("--max-batches", "max_batches")):
        if key in col:
            argv += [flag, str(col[key])]
    log = work / "collect.log"
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable] + argv, stdout=f,
                           stderr=subprocess.STDOUT, env=env,
                           cwd=_HERE.parent.parent)
    if p.returncode != 0:
        raise RuntimeError(f"collection driver exited {p.returncode} "
                           f"(see {log}); exit 2 = engine failure, "
                           f"exit 3 = health gate")
    return {"driver": driver, "log": str(log)}


def step_assemble(cfg: dict, st: State, work: Path, force: bool) -> dict:
    shards = Path(cfg["corpus"]["shards"])
    if (shards / "shards_index.json").is_file():
        return {"skipped": "shards_index.json exists"}
    dump_root = cfg["corpus"].get("dump_root")
    if not dump_root:
        raise RuntimeError("no assembled corpus and no corpus.dump_root "
                           "to assemble from")
    idx = dataset.assemble_shards(dump_root, shards, store_logits=False,
                                  compress=True)
    return {"num_blocks": idx["num_blocks"],
            "num_sequences": idx["num_sequences"]}


def step_geometry(cfg: dict, st: State, work: Path, force: bool) -> dict:
    from elb_train import routing_reader
    shards = Path(cfg["corpus"]["shards"])
    idx = dataset.load_index(shards)
    info = {k: idx[k] for k in ("n_experts", "topk", "max_gamma",
                                "num_blocks", "num_sequences")}
    info["n_moe_layers"] = len(idx["moe_layers"])
    routing = cfg["corpus"].get("routing")
    if routing:
        files = sorted(Path(routing).glob("*.bin"))
        if not files:
            raise RuntimeError(f"no routing .bin files under {routing}")
        v = routing_reader.open_run(files[0])
        if v.layers.tolist() != idx["moe_layers"]:
            raise RuntimeError("routing layer set != shards moe_layers")
        info["routing_files"] = len(files)
    # derive the deep-layer window when unset: the last ~27% of MoE
    # layers (matches GLM-5.2's L58-77 over layers 3..77).
    ml = idx["moe_layers"]
    s0 = cfg["stage0"]
    if s0["deep_lo"] is None or s0["deep_hi"] is None:
        n_deep = max(1, round(len(ml) * 0.27))
        s0["deep_lo"], s0["deep_hi"] = int(ml[-n_deep]), int(ml[-1])
        info["deep_window_derived"] = [s0["deep_lo"], s0["deep_hi"]]
    return info


def step_sidecars(cfg: dict, st: State, work: Path, force: bool) -> dict:
    shards = Path(cfg["corpus"]["shards"])
    info = {}
    if not (shards / "prev_top_ids_00000.npy").is_file() or force:
        info["prev"] = dataset.build_prev_membership_sidecars(shards)
    else:
        info["prev"] = "present"
    if not (shards / "recur_00000.npz").is_file() or force:
        info["recur"] = recurrence.build_recurrence_sidecars(shards)
    else:
        info["recur"] = "present"
    return info


def step_split(cfg: dict, st: State, work: Path, force: bool) -> dict:
    sp = cfg["split"]
    if sp["seed"] is not None:
        return {"seed": sp["seed"], "source": "pinned"}
    if not sp.get("select_if_unset"):
        raise RuntimeError("split.seed unset and select_if_unset false")
    out = work / "split_seed_selection.json"
    _run_cli(["tools/elb_train/select_split_seed.py",
              "--shards", str(cfg["corpus"]["shards"]),
              "--held-out-fraction", str(sp["held_fraction"]),
              "--min-held-sequences", str(sp["min_held_sequences"]),
              "--seeds", str(sp["candidate_seeds"]),
              "--json-out", str(out)], work / "split.log")
    sel = json.load(open(out))
    sp["seed"] = int(sel["chosen_seed"])
    return {"seed": sp["seed"], "source": "selected",
            "selection": str(out)}


def step_bars(cfg: dict, st: State, work: Path, force: bool) -> dict:
    bars = work / "bars_b0prev.json"
    if bars.is_file() and not force:
        return {"skipped": "present", "path": str(bars)}
    s0 = cfg["stage0"]
    _run_cli(["tools/elb_train/compute_b0_bars.py",
              "--shards", str(cfg["corpus"]["shards"]),
              "--split-seed", str(cfg["split"]["seed"]),
              "--held-fraction", str(cfg["split"]["held_fraction"]),
              "--deep-lo", str(s0["deep_lo"]),
              "--deep-hi", str(s0["deep_hi"]),
              "--out", str(bars)], work / "bars.log")
    b = json.load(open(bars))
    return {"path": str(bars), "b0prev_recall8": b["b0prev_recall8"],
            "prev_union_deep": b["prev_union_deep"]}


def step_router_bank(cfg: dict, st: State, work: Path, force: bool) -> dict:
    bank_npz = work / "router_bank.npz"
    if bank_npz.is_file() and not force:
        return {"skipped": "present", "path": str(bank_npz)}
    r = cfg["router"]
    if not r.get("checkpoint"):
        raise RuntimeError("router.checkpoint unset")
    idx = dataset.load_index(cfg["corpus"]["shards"])
    bank = glm_router.RouterBank.from_checkpoint(
        r["checkpoint"], idx["moe_layers"],
        weight_tmpl=r["weight_tmpl"], bias_tmpl=r["bias_tmpl"])
    bank.save_npz(bank_npz)
    return {"path": str(bank_npz),
            "shape": list(bank.weight.shape)}


def _stage0_argv(cfg: dict, work: Path, do: str, out: Path) -> list[str]:
    s0 = cfg["stage0"]
    return ["tools/elb_train/stage0.py",
            "--shards", str(cfg["corpus"]["shards"]),
            "--routing", str(cfg["corpus"]["routing"]),
            "--router-ckpt", str(cfg["router"]["checkpoint"]),
            "--split-seed", str(cfg["split"]["seed"]),
            "--held-fraction", str(cfg["split"]["held_fraction"]),
            "--do", do,
            "--bars", str(work / "bars_b0prev.json"),
            "--windows", s0["windows"], "--sigma-c", s0["sigma_c"],
            "--pool-m", s0["pool_m"],
            "--deep-lo", str(s0["deep_lo"]),
            "--deep-hi", str(s0["deep_hi"]),
            "--sigma-seqs", str(s0["sigma_seqs"]),
            "--evict-max-seqs", str(s0["evict_max_seqs"]),
            "--evict-caps", s0["evict_caps"],
            "--workdir", str(work), "--out", str(out)]


def step_verify(cfg: dict, st: State, work: Path, force: bool) -> dict:
    out = work / "verify.json"
    _run_cli(_stage0_argv(cfg, work, "verify", out), work / "verify.log")
    v = json.load(open(out))["verify"]
    if v["rank_set_agreement"] < cfg["gates"]["rank_set_agreement_min"]:
        raise RuntimeError(f"rank-set agreement {v['rank_set_agreement']} "
                           f"below gate")
    if abs(v.get("bar_delta", 0.0)) > cfg["gates"]["bar_tolerance"]:
        raise RuntimeError(f"b0_prev bar delta {v['bar_delta']} exceeds "
                           f"tolerance")
    if v["join_mismatch_cells"] != 0:
        raise RuntimeError("routing<->shard join mismatches")
    return v


def step_evaluate(cfg: dict, st: State, work: Path, force: bool) -> dict:
    out = work / "eval.json"
    _run_cli(_stage0_argv(cfg, work, "eval", out), work / "eval.log")
    r = json.load(open(out))
    d = r["decompose"]["deep"]
    return {"recall8_deep": d["recall8"], "miss_mass_deep": d["miss_mass"],
            "out": str(out)}


def step_evict(cfg: dict, st: State, work: Path, force: bool) -> dict:
    out = work / "evict.json"
    _run_cli(_stage0_argv(cfg, work, "evict", out), work / "evict.log")
    return {"out": str(out)}


def evaluate_verdict(eval_results: dict, verify: dict, gates: dict,
                     headline_window: str, headline_c: float) -> dict:
    """Pure §3-P11(6) verdict logic (unit-testable): the kill gate is the
    deep-layer within-pool absolute recall headroom at the headline cell;
    below the bar => (i) ~= 0 => P11 STOPS."""
    key = f"c{headline_c:g}|{headline_window}"
    grid = eval_results["decompose"]["deep"]["grid"]
    if key not in grid:
        raise KeyError(f"headline cell {key!r} not in decomposition grid "
                       f"({sorted(grid)})")
    cell = grid[key]
    headroom = cell["within_headroom"]
    gate_pass = headroom >= gates["within_headroom_min"]
    bank = eval_results.get("bank", {})
    ranked = sorted(
        ((n, r.get("S_recall8_all", 0.0)) for n, r in bank.items()),
        key=lambda t: -t[1])
    return {
        "verdict": "CONTINUE" if gate_pass else "STOP",
        "headline_cell": key,
        "within_headroom": headroom,
        "gate_bar": gates["within_headroom_min"],
        "f_boundary": cell["f_boundary"],
        "f_within": cell["f_within"],
        "f_novel": cell["f_novel"],
        "recall8_deep": eval_results["decompose"]["deep"]["recall8"],
        "recall_ceiling_deep":
            eval_results["decompose"]["deep"]["recall8"] + headroom,
        "bank_ranked": ranked[:5],
        "verify_ok": bool(verify.get("verify_ok", False)),
    }


def step_verdict(cfg: dict, st: State, work: Path, force: bool) -> dict:
    from elb_train import stage0 as stage0_mod
    ev = json.load(open(work / "eval.json"))
    verify = json.load(open(work / "verify.json"))["verify"]
    merged = dict(ev)
    merged["verify"] = verify
    evict_path = work / "evict.json"
    if evict_path.is_file():
        merged["evict"] = json.load(open(evict_path)).get("evict")
    s0 = cfg["stage0"]
    verdict = evaluate_verdict(ev, verify, cfg["gates"],
                               s0["headline_window"],
                               float(s0["headline_c"]))
    verdict["git_rev"] = _git_rev()
    verdict["config_name"] = cfg.get("name", "stage0")
    merged["verdict"] = verdict
    json.dump(merged, open(work / "results.json", "w"), indent=1)
    stage0_mod.write_table(merged, work / "results_table.md")
    study = cfg.get("study_dir")
    if study:
        study = Path(study)
        study.mkdir(parents=True, exist_ok=True)
        for fn in ("results.json", "results_table.md", "sigma.json"):
            src = work / fn
            if src.is_file():
                (study / fn).write_bytes(src.read_bytes())
        json.dump(verdict, open(study / "verdict.json", "w"), indent=1)
    print(f"[pipeline] VERDICT: {verdict['verdict']} — within-pool "
          f"headroom {verdict['within_headroom']:.4f} vs bar "
          f"{verdict['gate_bar']}", flush=True)
    return verdict


# ── P11.b (stageb) steps — gated on the Stage-0 CONTINUE verdict ─────────────

def _stageb_gate(cfg: dict, work: Path) -> dict | None:
    """Returns a skip-info dict when stageb must not run, else None."""
    if not cfg["stageb"].get("enabled", True):
        return {"skipped": "stageb.enabled is false"}
    vpath = work / "results.json"
    if not vpath.is_file():
        raise RuntimeError("stageb requires the Stage-0 verdict step")
    verdict = json.load(open(vpath)).get("verdict", {})
    if verdict.get("verdict") != "CONTINUE":
        return {"skipped": f"Stage-0 verdict is "
                           f"{verdict.get('verdict', 'missing')} — "
                           f"P11.b is gated off (§3-P11(6))"}
    return None


def _stageb_argv(cfg: dict, work: Path, do: str) -> list[str]:
    sb = cfg["stageb"]
    return ["tools/elb_train/stageb.py",
            "--shards", str(cfg["corpus"]["shards"]),
            "--routing", str(cfg["corpus"]["routing"]),
            "--router-ckpt", str(cfg["router"]["checkpoint"]),
            "--split-seed", str(cfg["split"]["seed"]),
            "--held-fraction", str(cfg["split"]["held_fraction"]),
            "--do", do,
            "--stride", str(sb["stride"]),
            "--jobs", str(sb.get("jobs", 0)),
            "--fit-cap", str(sb["fit_cap"]),
            "--gbm-rounds", str(sb["gbm_rounds"]),
            "--sigma", str(work / "sigma.json"),
            "--deep-lo", str(cfg["stage0"]["deep_lo"]),
            "--deep-hi", str(cfg["stage0"]["deep_hi"]),
            "--evict-max-seqs", str(sb["evict_max_seqs"]),
            "--evict-caps", sb["evict_caps"],
            "--workdir", str(work / "stageb"),
            "--out", str(work / "stageb" / "results.json")]


def _make_stageb_step(do: str):
    def step(cfg: dict, st: State, work: Path, force: bool) -> dict:
        gate = _stageb_gate(cfg, work)
        if gate is not None:
            return gate
        (work / "stageb").mkdir(exist_ok=True)
        _run_cli(_stageb_argv(cfg, work, do),
                 work / f"stageb_{do}.log")
        r = json.load(open(work / "stageb" / "results.json"))
        if do == "verdict":
            v = r["verdict"]
            study = cfg["stageb"].get("study_dir") or (
                cfg.get("study_dir") and
                str(Path(cfg["study_dir"]).parent
                    / (Path(cfg["study_dir"]).name + "-b")))
            if study:
                sd = Path(study)
                sd.mkdir(parents=True, exist_ok=True)
                (sd / "results.json").write_bytes(
                    (work / "stageb" / "results.json").read_bytes())
                json.dump(v, open(sd / "verdict.json", "w"), indent=1)
            print(f"[pipeline] stageb VERDICT: capacity_candidate_enters"
                  f" = {v['capacity_candidate_enters']} (gbdt gate "
                  f"{v['gbdt_gate_fires']}, linear-leaves-skill "
                  f"{v['linear_leaves_residual_skill']})", flush=True)
            return v
        return {"log": str(work / f'stageb_{do}.log')}
    return step


def _make_stagec_step(do: str):
    def step(cfg: dict, st: State, work: Path, force: bool) -> dict:
        sc = cfg["stagec"]
        if not sc.get("enabled", True):
            return {"skipped": "stagec.enabled is false"}
        bres = work / "stageb" / "results.json"
        if not bres.is_file() or "verdict" not in json.load(open(bres)):
            return {"skipped": "stageb verdict missing — run the "
                               "stageb steps first (P11.c consumes the "
                               "fitted probe)"}
        cwork = work / "stagec"
        cwork.mkdir(exist_ok=True)
        argv = ["tools/elb_train/stagec.py",
                "--shards", str(cfg["corpus"]["shards"]),
                "--routing", str(cfg["corpus"]["routing"]),
                "--split-seed", str(cfg["split"]["seed"]),
                "--held-fraction", str(cfg["split"]["held_fraction"]),
                "--do", do,
                "--jobs", str(sc.get("jobs", 0)),
                "--device", sc.get("device", ""),
                "--sigma", str(work / "sigma.json"),
                "--deep-lo", str(cfg["stage0"]["deep_lo"]),
                "--deep-hi", str(cfg["stage0"]["deep_hi"]),
                "--evict-max-seqs", str(sc["evict_max_seqs"]),
                "--evict-caps", sc["evict_caps"],
                "--stageb-workdir", str(work / "stageb"),
                "--workdir", str(cwork)]
        _run_cli(argv, work / f"stagec_{do}.log")
        r = json.load(open(cwork / "results.json"))
        if do == "verdict":
            v = r["verdict"]
            study = sc.get("study_dir")
            if study:
                sd = Path(study)
                sd.mkdir(parents=True, exist_ok=True)
                (sd / "results.json").write_bytes(
                    (cwork / "results.json").read_bytes())
                json.dump(v, open(sd / "verdict.json", "w"), indent=1)
            print(f"[pipeline] stagec VERDICT: combined_beats_free = "
                  f"{v['combined_beats_free']}, pool32 "
                  f"{v['pool32_combined']:.4f}", flush=True)
            return v
        return {"log": str(work / f'stagec_{do}.log')}
    return step


def _step_cpp_fixture(cfg: dict, st: State, work: Path,
                      force: bool) -> dict:
    """E2E contract for the C++ ExpertRidge inference seam: every
    artifact the C++ module and its parity test consume is produced
    by this pipeline — the published model dirs come from the run/
    package step, and this step regenerates the golden parity
    fixture (deterministic small-dims replay of the ship semantics
    through the Python engine, single-file safetensors readable by
    the engine's SafetensorsReader)."""
    sc = cfg["stagecprime"]
    if not sc.get("enabled", False):
        return {"skipped": "stagecprime.enabled is false"}
    out = sc.get("cpp_fixture_out", "")
    if not out:
        return {"skipped": "cpp_fixture_out empty"}
    argv = ["tools/elb_train/export_cpp_fixture.py", str(out)]
    _run_cli(argv, work / "stagecprime_cppfix.log")
    return {"fixture": str(out),
            "bytes": Path(out).stat().st_size}


def _make_stagecprime_step(do: str):
    def step(cfg: dict, st: State, work: Path, force: bool) -> dict:
        sc = cfg["stagecprime"]
        if not sc.get("enabled", False):
            return {"skipped": "stagecprime.enabled is false"}
        if not (work / "stageb" / "results.json").is_file():
            return {"skipped": "stageb results missing — the c' engine "
                               "consumes the fitted probe"}
        cwork = work / "stagecprime"
        cwork.mkdir(exist_ok=True)
        eff_do = do
        publish = sc.get("publish_model_dirs", "") \
            or sc.get("publish_model_dir", "")
        if do == "run" and publish:
            eff_do = "run,package"      # model dirs = final run output
        argv = ["tools/elb_train/stagecprime.py",
                "--shards", str(cfg["corpus"]["shards"]),
                "--routing", str(cfg["corpus"]["routing"]),
                "--split-seed", str(cfg["split"]["seed"]),
                "--held-fraction", str(cfg["split"]["held_fraction"]),
                "--do", eff_do,
                "--package-exp", sc.get("champion_exp", "x7g_memotok"),
                "--exp", sc.get("exps", "all"),
                "--device", sc.get("device", "cuda:0"),
                "--router-ckpt", str(cfg["router"]["checkpoint"]),
                "--embed-tensor", sc.get("embed_tensor",
                                         "model.embed_tokens.weight"),
                "--memo-buckets", str(sc.get("memo_buckets", "4096")),
                "--sigma", str(work / "sigma.json"),
                "--deep-lo", str(cfg["stage0"]["deep_lo"]),
                "--deep-hi", str(cfg["stage0"]["deep_hi"]),
                "--stageb-workdir", str(work / "stageb"),
                "--workdir", str(cwork)]
        if sc.get("export_configs"):
            argv += ["--export-configs", sc["export_configs"]]
        if sc.get("publish_model_dirs"):
            argv += ["--publish-map", sc["publish_model_dirs"]]
        elif publish:
            argv += ["--package-dir", str(publish)]
        if do == "run" and not force:
            argv.append("--skip-done")      # pipeline_state resume
        _run_cli(argv, work / f"stagecprime_{do.replace(',', '_')}.log")
        r = json.load(open(cwork / "results.json"))
        if do == "verdict":
            v = r["verdict"]
            study = sc.get("study_dir")
            if study:
                sd = Path(study)
                sd.mkdir(parents=True, exist_ok=True)
                (sd / "results.json").write_bytes(
                    (cwork / "results.json").read_bytes())
                json.dump(v, open(sd / "verdict.json", "w"), indent=1)
            print(f"[pipeline] stagecprime VERDICT: best "
                  f"{v['best']['experiment']} pool32 "
                  f"{v['best']['pool32']}", flush=True)
            return v
        return {"log": str(work / f'stagecprime_{do}.log')}
    return step


STEP_FNS = {"collect": step_collect, "assemble": step_assemble,
            "geometry": step_geometry, "sidecars": step_sidecars,
            "split": step_split, "bars": step_bars,
            "router_bank": step_router_bank, "verify": step_verify,
            "evaluate": step_evaluate, "evict": step_evict,
            "verdict": step_verdict,
            "stageb_dump": _make_stageb_step("dump"),
            "stageb_fit": _make_stageb_step("fit"),
            "stageb_eval": _make_stageb_step("eval"),
            "stageb_evict": _make_stageb_step("evict"),
            "stageb_verdict": _make_stageb_step("verdict"),
            "stagec_run": _make_stagec_step("run"),
            "stagec_evict": _make_stagec_step("evict"),
            "stagec_verdict": _make_stagec_step("verdict"),
            "stagecprime_prep": _make_stagecprime_step(
                "prep-kmeans,prep-tap"),
            "stagecprime_run": _make_stagecprime_step("run"),
            "stagecprime_verdict": _make_stagecprime_step("verdict"),
            "stagecprime_cppfix": _step_cpp_fixture}


# ── runner ───────────────────────────────────────────────────────────────────

def run(cfg: dict, from_step: str | None = None, to_step: str | None = None,
        only: str | None = None, force: bool = False) -> int:
    work = Path(cfg["workdir"])
    work.mkdir(parents=True, exist_ok=True)
    st = State(work)
    st.data.setdefault("git_rev", _git_rev())
    st.data["config_snapshot"] = cfg
    st.save()
    if only:
        selected = [only]
    else:
        lo = STEP_ORDER.index(from_step) if from_step else 0
        hi = STEP_ORDER.index(to_step) if to_step else len(STEP_ORDER) - 1
        selected = list(STEP_ORDER[lo:hi + 1])
    for name in selected:
        rec = st.step(name)
        rerun = force or (from_step and name == from_step) or only
        if rec.get("status") == "done" and not rerun:
            print(f"[pipeline] {name}: done (cached)", flush=True)
            continue
        print(f"[pipeline] {name}: running ...", flush=True)
        st.mark(name, "running")
        try:
            info = STEP_FNS[name](cfg, st, work, bool(force))
        except Exception as e:  # noqa: BLE001 — step boundary
            st.mark(name, "failed", {"error": str(e)})
            print(f"[pipeline] {name}: FAILED — {e}", flush=True)
            return 1
        status = "skipped" if isinstance(info, dict) and "skipped" in info \
            else "done"
        st.mark(name, status, info)
        print(f"[pipeline] {name}: {status}", flush=True)
        # split step may have resolved the seed — persist for resume.
        st.data["config_snapshot"] = cfg
        st.save()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--from", dest="from_step", choices=STEP_ORDER)
    ap.add_argument("--to", dest="to_step", choices=STEP_ORDER)
    ap.add_argument("--only", choices=STEP_ORDER)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print step status and exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.list:
        st = State(Path(cfg["workdir"]))
        for name in STEP_ORDER:
            rec = st.step(name)
            print(f"{name:12s} {rec.get('status', '-'):8s} "
                  f"{rec.get('ended', '')}")
        return 0
    return run(cfg, a.from_step, a.to_step, a.only, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
