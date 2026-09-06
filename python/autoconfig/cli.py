"""Autoconfig CLI (spec/AUTO_RUN.md, AUTOCONFIG §1/§7).

Two entry points, one solver:

  auto-run (what a user runs)   ``--model <weights>`` — probe the model,
      calibrate the box, derive the recipe, train the loader constants on a
      real 100-token decode, print the serve command. Everything else has a
      good default; the four levers are optional flags.

  legacy/advanced               ``--config <recipe>`` — derive from an
      existing recipe's identity (this is what ``serve.py --autoconfig``
      calls, and what re-derives a champion-shaped config in place).

Re-derivation happens only on hardware-fingerprint change or --redetect;
measured artifacts are reused unless --reset/--retrain ask otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .benchmark import BenchmarkError, BenchmarkOptions
from .compare import compare, render_report
from . import templates
from .calibration import load_calibration, accept_calibration, \
    resolve_calibration_path
from .enginerun import EngineRunError
from .explain import Infeasible
from .fingerprint import fingerprint
from .hwdetect import detect_hardware, HardwareDescriptor
from .modelshape import ModelShape
from .pipeline import (AutoRunOptions, expert_slot_bytes, pinned_bytes,
                       run_auto, run_benchmark_mode, serve_command,
                       solve_recipe)
from .pins import parse_pin_args
from .solver import ACCURACY_TIERS, AutoconfigKnobs, Levers, \
    PREFERENCES


def _knobs_for(base: dict, override):
    if override is not None:
        return override
    from .config import knobs_from_config
    return knobs_from_config(base)


def default_output_path(base_config_path: str) -> str:
    stem, _ = os.path.splitext(base_config_path)
    return stem + ".autoconfig.json"


# Kept as module-level names: the byte-sizing helpers moved to pipeline.py
# when the auto-run pipeline needed them too.
_expert_slot_bytes = expert_slot_bytes
_pinned_bytes = pinned_bytes


def run(base_config_path: str, out_path: str = "",
        vram_expert_ratio: float | None = None,
        total_active_context_tokens: int | None = None,
        redetect: bool = False, repo_root: str = ".",
        hardware_json: str = "", compare_to: str = "",
        knobs: "AutoconfigKnobs | None" = None,
        prefer: str | None = None,
        accuracy: str | None = None,
        pins=None, base_override: "dict | None" = None) -> int:
    """Derive from a BASE RECIPE (the identity source). Unchanged contract —
    serve.py --autoconfig and the champion acceptance test call this.

    ``base_override`` (P-31): an in-memory identity base (the probe-built
    one from pipeline.base_from_source) instead of a file — what
    ``serve.py --autoconfig --model`` uses; ``out_path`` is then required
    (there is no file to name a default after)."""
    if base_override is not None:
        base = base_override
        if not out_path:
            raise ValueError("base_override requires an explicit out_path")
    else:
        with open(base_config_path) as f:
            base = json.load(f)
    knobs = _knobs_for(base, knobs)
    out_path = out_path or (base.get("autoconfig") or {}).get("output_path") \
        or default_output_path(base_config_path)
    lv = Levers(
        vram_expert_ratio if vram_expert_ratio is not None
        else (base.get("autoconfig") or {}).get("vram_expert_ratio"),
        total_active_context_tokens if total_active_context_tokens is not None
        else (base.get("autoconfig") or {}).get("total_active_context_tokens"),
        # NOTE: read from the base recipe, never WRITTEN back into it —
        # `autoconfig` is additionalProperties:false in config/schema.json and
        # does not carry the field yet (schema work is serialised in this
        # campaign). The read side is live so adding the schema property is
        # the only remaining edit.
        prefer if prefer is not None
        else ((base.get("autoconfig") or {}).get("prefer") or "balanced"),
        accuracy if accuracy is not None
        else ((base.get("autoconfig") or {}).get("accuracy") or "standard"))

    if hardware_json:
        with open(hardware_json) as f:
            hw = HardwareDescriptor.from_json(json.load(f))
        src = f"descriptor file {hardware_json}"
    else:
        hw = detect_hardware()
        src = "live /proc + /sys"
    if not hw.gpus:
        print("autoconfig: no NVIDIA GPUs detected — nothing to fit", file=sys.stderr)
        return 2

    # P-29 step 13: whole-recipe form — the MTP expert census is armed by the
    # SPECULATION section (see modelshape.mtp_experts_armed).
    shape = ModelShape.from_config(base)
    cal = None
    rej: list = []
    cal_path = resolve_calibration_path(
        (base.get("gpu_loader") or {}).get("calibration_path", ""),
        os.path.join(repo_root, (base.get("model") or {}).get("weights_path", "")))
    if os.path.exists(cal_path):
        cal = load_calibration(cal_path)
        order = [g.ordinal for g in sorted(hw.gpus, key=lambda g: (-g.vram_mib, g.ordinal))]
        rej = accept_calibration(cal, hw, order, shape.hidden_size,
                                 shape.moe_intermediate_size)
        if rej:
            print(f"autoconfig: calibration {cal_path} REJECTED: {'; '.join(rej)}",
                  file=sys.stderr)
            cal = None

    fp = fingerprint(hw, cal)
    if os.path.exists(out_path) and not redetect and not pins:
        # pins never bypass derivation: reusing a stored config would
        # silently ignore a HARD constraint (design §2 corollary)
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if (existing.get("autoconfig") or {}).get("fingerprint") == fp:
                # TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS: the reuse
                # key is hardware fingerprint + LEVERS — a lever the caller
                # passed must change the output or be refused, never be
                # silently discarded by a fingerprint-only cache (three
                # --active-context values once returned the same recipe).
                ex_ac = existing.get("autoconfig") or {}
                stored_lv = Levers(
                    ex_ac.get("vram_expert_ratio"),
                    ex_ac.get("total_active_context_tokens"),
                    ex_ac.get("prefer") or "balanced",
                    ex_ac.get("accuracy") or "standard")
                if stored_lv == lv:
                    print(f"autoconfig: {out_path} is current for this "
                          f"hardware ({fp}) AND these levers — reusing "
                          f"(pass --redetect to force)")
                    return 0
                print(f"autoconfig: {out_path} matches this hardware ({fp}) "
                      f"but NOT the requested levers (stored {stored_lv} vs "
                      f"requested {lv}) — re-deriving")
        except (OSError, ValueError):
            pass

    # TD-AUTOCONFIG-DRAFT-IDENTITY: the legacy path must be as loud as the
    # auto-run path (pipeline.py) about a recipe-named draft that is simply
    # not on this box — deriving draft-less SILENTLY is the defect, and the
    # warning belongs on stderr AND in the explain sidecar.
    draft_missing_warning = ""
    _ck = ((base.get("speculation") or {}).get("dspark") or {}).get(
        "checkpoint_path", "")
    if _ck:
        from .solver import load_draft_candidate as _ldc
        if _ldc(_ck, repo_root) is None:
            draft_missing_warning = (
                f"speculation.dspark.checkpoint_path {_ck} does not exist on "
                f"this box — deriving WITHOUT a draft (the recipe asked for "
                f"one; put the checkpoint there or drop the speculation "
                f"section to silence this)")
            print("autoconfig WARNING: " + draft_missing_warning,
                  file=sys.stderr)

    try:
        res = solve_recipe(base, hw, lv, knobs, repo_root, cal=cal,
                           cal_reject=tuple(rej), pins=pins)
    except Infeasible as e:
        print(str(e), file=sys.stderr)
        return 3

    res.recipe["autoconfig"] = {
        "enabled": True,
        "vram_expert_ratio": lv.vram_expert_ratio,
        "total_active_context_tokens": lv.total_active_context_tokens,
        "output_path": out_path,
        "fingerprint": fp,
    }
    # Schema-backed levers (TD-KVXP-SCHEMA-KNOBS window, 2026-09-02):
    # carried ONLY when non-default, so default derivations stay
    # byte-identical while a lever-carrying recipe re-applies its
    # preference on every --autoconfig boot.  Pins stay CLI-only by
    # design (a self-pinned recipe would re-apply stale hard constraints).
    if lv.prefer != "balanced":
        res.recipe["autoconfig"]["prefer"] = lv.prefer
    if lv.accuracy != "standard":
        res.recipe["autoconfig"]["accuracy"] = lv.accuracy
    res.explanations.add("autoconfig.fingerprint", fp, "measured",
                         f"measured-topology fingerprint over {src}"
                         + (" + calibration tier matrix" if cal else
                            " (no calibration — speed choices provisional)"))

    with open(out_path, "w") as f:
        json.dump(res.recipe, f, indent=2)
        f.write("\n")
    explain_path = out_path.rsplit(".json", 1)[0] + ".explain.md"
    if draft_missing_warning and draft_missing_warning not in res.warnings:
        res.warnings.append(draft_missing_warning)

    with open(explain_path, "w") as f:
        f.write(res.explanations.render_markdown(
            f"Autoconfig derivation — {os.path.basename(out_path)}"))
        if res.warnings:
            f.write("\n## Warnings\n\n")
            for w in res.warnings:
                f.write(f"- {w}\n")
        if res.degradations:
            f.write("\n## Degradations (E1)\n\n")
            for d in res.degradations:
                f.write(f"- {d}\n")
    for line in res.explanations.log_lines():
        print(line)
    for w in res.warnings:
        print(f"autoconfig WARNING: {w}", file=sys.stderr)
    print(f"autoconfig: wrote {out_path} (+ {os.path.basename(explain_path)})")

    if compare_to:
        with open(compare_to) as f:
            ref = json.load(f)
        divs = compare(ref, res.recipe, res.explanations,
                       template=templates.template_view(
                           (res.recipe.get("model") or {})
                           .get("architecture", "")))
        report = render_report(os.path.basename(compare_to), divs)
        report_path = out_path.rsplit(".json", 1)[0] + ".compare.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        if any(d.klass == "unexplained" for d in divs):
            return 4
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="autoconfigure",
        description="Point it at a model; it fits this box and prints the "
                    "command that serves it (spec/AUTO_RUN.md). Opt-in: a "
                    "hand-tuned recipe always wins unless you run this.")
    ap.add_argument("--model", default="",
                    help="THE model: a .gguf file, a directory of GGUF "
                         "shards, or a HuggingFace model directory")
    ap.add_argument("--draft", default="",
                    help="speculative draft checkpoint (.dspark). Default: "
                         "discover one beside the model; 'none' serves "
                         "without a draft")
    ap.add_argument("--vram-expert-ratio", type=float, default=None,
                    help="lever 1 (optional): VRAM-resident expert slots / "
                         "all expert slots. Default: fill the expert hosts")
    ap.add_argument("--active-context", type=int, default=None,
                    help="lever 2 (optional): TOTAL active context tokens "
                         "across concurrent requests, not the per-request "
                         "max. Default: 2 x 64k")
    ap.add_argument("--prefer", choices=PREFERENCES, default=None,
                    help="lever 3 (optional): what the fit gives up FIRST "
                         "when the box is tight. 'speed' sheds concurrency "
                         "and context before it buys a measured slowdown; "
                         "'capacity' buys the slowdown to protect your ask; "
                         "'balanced' (default) is the shipped ladder. Order "
                         "only — it never changes what is feasible")
    ap.add_argument("--pin", action="append", default=None, metavar="PIN",
                    help="HARD constraint the solver treats as fixed and "
                         "solves around (repeatable): a JSON file of config "
                         "paths to values, or an inline dotted.path=value "
                         "(e.g. compute.attention_backend=snapmla). NOT a "
                         "lever: it survives every degradation rung, and an "
                         "infeasible pin set REFUSES naming the binding pin "
                         "— never quietly relaxed. Pinnable = the solver's "
                         "decision surface only; the .explain.md marks "
                         "pinned fields 'pinned' (told, not derived); a pin "
                         "contradicting a measured registry row derives "
                         "LOUDLY (spec/plans/"
                         "AUTOCONFIG_PINNED_CONSTRAINTS.md)")
    ap.add_argument("--accuracy", choices=ACCURACY_TIERS, default=None,
                    help="lever 4 (optional): the numerics FLOOR. "
                         "'standard' (default) keeps the family template's "
                         "champion-proven backend; 'high' floors at the "
                         "all-FP8 KV path (no 4-bit TQ codec) and pays the "
                         "KV bytes; 'compact' floors at the most "
                         "byte-efficient serveable backend; 'superior' "
                         "(full-precision KV, no TQ) is DEFINED BUT NOT "
                         "IMPLEMENTED — it refuses, naming the missing "
                         "kFull codec arm. A floor over measured options, "
                         "never a weight")
    ap.add_argument("--name", default="",
                    help="base name for the generated config and artifacts "
                         "(default: the model's own name)")
    ap.add_argument("--prefix", default="",
                    help="prefix prepended to the default base name")
    ap.add_argument("--out-dir", default="",
                    help="where the config and its sidecars are written "
                         "(default: the current directory)")
    ap.add_argument("--reset", action="store_true",
                    help="start from scratch: re-measure the calibration AND "
                         "re-train it, overwriting both")
    ap.add_argument("--retrain", action="store_true",
                    help="keep the hardware calibration, re-run the "
                         "100-token training pass")
    ap.add_argument("--skip-training", action="store_true",
                    help="stop after deriving the config (serve on the "
                         "uncorrected hardware calibration)")
    ap.add_argument("--calibration", default="",
                    help="use THIS hardware-calibration artifact instead of "
                         "measuring one")
    ap.add_argument("--trained", default="",
                    help="use THIS trained artifact instead of running the "
                         "training pass")
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer directory (default: discovered beside "
                         "the weights)")
    ap.add_argument("--prepacked", default="",
                    help="prepacked expert store (default: discovered by "
                         "manifest identity)")
    ap.add_argument("--placement-table", default="",
                    help="use THIS arena placement table (CSV) instead of "
                         "fitting one in step 4")
    ap.add_argument("--refit-placement", action="store_true",
                    help="re-fit the arena placement table, capturing a "
                         "decode trace for it if step 3 did not run")
    ap.add_argument("--skip-placement", action="store_true",
                    help="do not fit a placement table (the online migrator "
                         "converges the layout on its own)")
    ap.add_argument("--allow-warm-holder", action="store_true",
                    help="calibrate even with an arena holder running (it "
                         "usually node-OOMs; kill the holder instead)")
    ap.add_argument("--config", default="",
                    help="base recipe: with --model it carries extras "
                         "(measured placement tables, serving overrides); "
                         "without --model it IS the identity source (legacy "
                         "mode, what serve.py --autoconfig uses)")
    ap.add_argument("--out", default="", help="legacy mode: output recipe path")
    ap.add_argument("--redetect", action="store_true",
                    help="re-derive even when the stored fingerprint matches "
                         "(keeps the measured artifacts; --reset re-measures)")
    ap.add_argument("--repo-root", default=".",
                    help="root for repo-relative paths")
    ap.add_argument("--hardware", default="",
                    help="hardware descriptor JSON (what-if / tests) instead "
                         "of live detection")
    ap.add_argument("--compare", default="",
                    help="reference recipe to diff + classify against")

    g = ap.add_argument_group(
        "benchmark mode",
        "Measure a recipe and EXIT (spec/AUTO_RUN.md 'benchmark'). "
        "--benchmark --config <recipe> measures an EXISTING config with no "
        "derivation; --model <weights> --benchmark runs the pipeline first "
        "and then measures what it produced. Either way one engine boots, "
        "both legs are measured, the engine is stopped and the results are "
        "written beside the recipe.")
    g.add_argument("--benchmark", action="store_true",
                   help="measure the recipe (prefill ladder + generation "
                        "tok/s), write the results, and exit")
    g.add_argument("--benchmark-label", default="",
                   help="arm name carried on every row, for charts that "
                        "compare several configs (default: the config's "
                        "file stem)")
    g.add_argument("--benchmark-ladder", default="",
                   help="explicit prompt lengths in tokens, comma-separated "
                        "(default: a halving ladder scaled to the recipe's "
                        "own serving.max_sequence_length)")
    g.add_argument("--benchmark-ladder-steps", type=int, default=4,
                   help="auto-ladder rungs (default 4)")
    g.add_argument("--benchmark-ladder-floor", type=int, default=1024,
                   help="auto-ladder smallest rung in tokens (default 1024)")
    g.add_argument("--benchmark-repeats", type=int, default=3,
                   help="repetitions per ladder rung (default 3 — the 20k "
                        "point has a documented +-1.3%% variance class, so "
                        "one sample is not a measurement)")
    g.add_argument("--benchmark-ladder-gen-tokens", type=int, default=8,
                   help="tokens each LADDER rung decodes after its prefill "
                        "(default 8). Raise it to measure DECODE AT CONTEXT: "
                        "the generation leg's prompt is short, so it reports "
                        "decode at ~0 context and cannot show how generation "
                        "scales with sequence length")
    g.add_argument("--benchmark-gen-runs", type=int, default=5,
                   help="generation repetitions (default 5, GF3.15's shape)")
    g.add_argument("--benchmark-gen-tokens", type=int, default=300,
                   help="tokens per generation run (default 300)")
    g.add_argument("--benchmark-corpus", default="",
                   help="prompt corpus: a whitespace-separated token-id file "
                        "(GATE_CORPUS format) or a text file. Default: this "
                        "checkout's own prose, tokenized with the recipe's "
                        "tokenizer")
    g.add_argument("--benchmark-out", default="",
                   help="results JSON path (default: <recipe>.benchmark.json "
                        "beside the recipe; the CSV and the ledger snippet "
                        "take the same stem)")
    g.add_argument("--benchmark-request-timeout", type=float, default=0.0,
                   help="per-request hang guard in seconds (default: derived "
                        "from the prompt length)")
    g.add_argument("--benchmark-skip-ladder", action="store_true",
                   help="measure generation only")
    g.add_argument("--benchmark-skip-generation", action="store_true",
                   help="measure the prefill ladder only")
    g.add_argument("--benchmark-allow-busy-box", action="store_true",
                   help="boot even though another serve process or GPU "
                        "compute app is live (dossier 1: NEVER two engines "
                        "— the second one OOM-kills the machine)")
    return ap


def benchmark_options(args) -> BenchmarkOptions:
    ladder = tuple(int(x) for x in args.benchmark_ladder.replace(",", " ")
                   .split()) if args.benchmark_ladder else ()
    return BenchmarkOptions(
        label=args.benchmark_label, ladder=ladder,
        ladder_steps=args.benchmark_ladder_steps,
        ladder_floor=args.benchmark_ladder_floor,
        repeats=args.benchmark_repeats,
        gen_runs=args.benchmark_gen_runs,
        gen_tokens=args.benchmark_gen_tokens,
        ladder_gen_tokens=args.benchmark_ladder_gen_tokens,
        corpus=args.benchmark_corpus, out_path=args.benchmark_out,
        request_timeout_s=args.benchmark_request_timeout,
        skip_ladder=args.benchmark_skip_ladder,
        skip_generation=args.benchmark_skip_generation,
        allow_busy_box=args.benchmark_allow_busy_box,
        tokenizer_path=args.tokenizer)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    pins = None
    if args.pin:
        try:
            pins = parse_pin_args(args.pin)
        except ValueError as e:
            print(f"autoconfigure: {e}", file=sys.stderr)
            return 2
    if args.benchmark and not args.model:
        # BENCHMARK an existing recipe: no derivation at all, so four
        # already-derived arms can each be measured without re-deriving.
        # With --benchmark, --config names the recipe to MEASURE (it is the
        # identity source for legacy derivation only when --benchmark is
        # absent).
        if not args.config:
            print("autoconfigure: --benchmark needs a recipe — pass "
                  "--config <recipe>.json to measure an existing one, or "
                  "--model <weights> to derive one and measure it",
                  file=sys.stderr)
            return 2
        if pins:
            print("autoconfigure: --pin does nothing when measuring an "
                  "EXISTING recipe (--benchmark --config runs no "
                  "derivation) — derive with the pins first, then "
                  "benchmark the result", file=sys.stderr)
            return 2
        try:
            rec = run_benchmark_mode(args.config, args.repo_root,
                                     benchmark_options(args))
        except Infeasible as e:
            print(str(e), file=sys.stderr)
            return 3
        except (BenchmarkError, EngineRunError) as e:
            print(f"autoconfig: benchmark FAILED — {e}", file=sys.stderr)
            return 6
        return 6 if rec.get("aborted") else 0
    if not args.model:
        if not args.config:
            print("autoconfigure: pass --model <weights> (or --config "
                  "<recipe> for legacy mode)", file=sys.stderr)
            return 2
        # Legacy derive-only mode honors: --out, the four levers, --pin,
        # --redetect, --repo-root, --hardware, --compare. Every auto-run
        # flag it would once have silently discarded now REFUSES
        # (TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS): a flag the
        # user passed must change the output or be rejected.
        ignored = [flag for flag, val in (
            ("--name", args.name), ("--prefix", args.prefix),
            ("--out-dir", args.out_dir), ("--draft", args.draft),
            ("--calibration", args.calibration),
            ("--trained", args.trained),
            ("--tokenizer", args.tokenizer),
            ("--prepacked", args.prepacked),
            ("--placement-table", args.placement_table),
            ("--reset", args.reset), ("--retrain", args.retrain),
            ("--skip-training", args.skip_training),
            ("--refit-placement", args.refit_placement),
            ("--skip-placement", args.skip_placement),
            ("--allow-warm-holder", args.allow_warm_holder)) if val]
        if ignored:
            print("autoconfigure: " + ", ".join(ignored) + " belong(s) to "
                  "the auto-run pipeline — legacy --config mode (no --model) "
                  "is derive-only and would silently discard "
                  + ("them" if len(ignored) > 1 else "it")
                  + ". Pass --model <weights> to run the pipeline (--config "
                  "then carries the base recipe), use --out for the legacy "
                  "output path, or drop the flag(s).", file=sys.stderr)
            return 2
        return run(args.config, args.out, args.vram_expert_ratio,
                   args.active_context, args.redetect, args.repo_root,
                   args.hardware, args.compare, prefer=args.prefer,
                   accuracy=args.accuracy, pins=pins)
    opts = AutoRunOptions(
        model_path=args.model, draft_path=args.draft,
        vram_expert_ratio=args.vram_expert_ratio,
        total_active_context_tokens=args.active_context,
        # None = unset (an unset lever never blocks reusing a current
        # config; an explicit one that differs from the stored derivation
        # forces a re-derive — TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-
        # LEVERS applies to the auto-run reuse gate too)
        prefer=args.prefer,
        accuracy=args.accuracy,
        name=args.name, prefix=args.prefix, out_dir=args.out_dir,
        repo_root=args.repo_root, reset=args.reset, retrain=args.retrain,
        skip_training=args.skip_training, calibration=args.calibration,
        trained=args.trained, base_config=args.config,
        compare_to=args.compare, hardware_json=args.hardware,
        tokenizer_path=args.tokenizer, prepacked_dir=args.prepacked,
        placement_table=args.placement_table, redetect=args.redetect,
        refit_placement=args.refit_placement,
        skip_placement=args.skip_placement,
        allow_warm_holder=args.allow_warm_holder, pins=pins)
    try:
        res = run_auto(opts)
    except Infeasible as e:
        print(str(e), file=sys.stderr)
        return 3
    except EngineRunError as e:
        print(f"autoconfig: measured step FAILED — {e}", file=sys.stderr)
        return 5
    if args.benchmark:
        # The pipeline finished and printed the serve command; now MEASURE
        # what it produced, on the same box, and exit.
        try:
            rec = run_benchmark_mode(res.config_path, args.repo_root,
                                     benchmark_options(args), res.log_dir)
        except Infeasible as e:
            print(str(e), file=sys.stderr)
            return 3
        except (BenchmarkError, EngineRunError) as e:
            print(f"autoconfig: benchmark FAILED — {e}", file=sys.stderr)
            return 6
        if rec.get("aborted"):
            return 6
    return 4 if res.unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
