"""The auto-run pipeline (spec/AUTO_RUN.md) — one command, three steps.

    python python/cli/autoconfigure.py --model <weights>

  step 1  CALIBRATE  measure this box's loader constants (engine boot), or
                     reuse an artifact that already matches it;
  step 2  DERIVE     solve the recipe (AUTOCONFIG §5) against the measured
                     hardware and the model's own shape;
  step 3  TRAIN      correct the constants against a REAL 100-token decode
                     (tools/loader_xray/trainer_apply.py) and point the
                     emitted recipe at the TRAINED artifact.
  step 4  PLACE      fit the host arena's per-(layer,expert) demand-fetch
                     table from step 3's own perf trace
                     (tools/loader_xray/freq_table.py) and carry it in the
                     recipe.

Ordering is not arbitrary: the trainer fits corrections to a BASELINE
calibration, so calibration must exist first; the training decode needs a
bootable recipe, so derivation must come before it; and the placement fit
consumes the trace that decode produces, so it comes last and costs no extra
boot when step 3 ran. The final recipe is the
one the user serves — it names the trained constants, and nothing about
serving re-runs any of this.

Reuse policy (the same for both measured artifacts): an artifact that exists
and matches this box + model is reused; ``--reset`` overwrites both from
scratch; ``--calibration`` / ``--trained`` supply one explicitly (which also
lets a box with no GPU time re-derive a config from artifacts it already has).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

from . import gguf_meta, modelprobe, sizing
from .calibration import (accept_calibration, load_calibration,
                          resolve_calibration_path)
from .compare import compare, render_report, summary_line
from . import templates
from .enginerun import EngineEnv, EngineRunError, run_calibration_boot, \
    run_capture_boot, run_freq_table_fit, run_trainer_apply, run_training_boot
from .explain import Infeasible
from .benchmark import (BenchmarkOptions, run_benchmark, summarize,
                        write_artifacts)
from .fingerprint import fingerprint
from .hwdetect import detect_hardware, HardwareDescriptor
from .modelshape import ModelShape, mtp_experts_armed
from .solver import (AutoconfigKnobs, Levers, Solver,
                     draft_identity_errors, load_draft_candidate)


# --------------------------------------------------------- model byte sizes

def expert_slot_bytes(base: dict, shape: ModelShape, repo_root: str):
    """The engine's expert slot unit, best source first: the prepack
    manifest (the engine's own number), then the GGUF tensor headers via the
    GG-9 per-projection MAX rule, then an analytic estimate (flagged)."""
    pre = (base.get("preprocessing") or {}).get("prepacked_dir", "")
    for root in (repo_root, "."):
        if pre:
            v = gguf_meta.expert_slot_bytes_from_manifest(os.path.join(root, pre))
            if v:
                return v, f"prepack manifest slot_size_bytes ({pre})"
    w = (base.get("model") or {}).get("weights_path", "")
    fmt = (base.get("model") or {}).get("weights_format", "")
    for root in (repo_root, "."):
        p = os.path.join(root, w)
        if fmt == "gguf" and os.path.exists(p):
            try:
                v, detail = gguf_meta.expert_slot_bytes_from_gguf(p)
                types = ",".join(f"{k}:{d['type']}" for k, d in sorted(detail.items()))
                return v, f"GGUF tensor headers, GG-9 per-projection max ({types})"
            except (ValueError, OSError):
                pass
    q = (base.get("quantization") or {}).get("weights", "")
    if q == "nvfp4":
        b = sum(sizing.nvfp4_projection_bytes(o, i) for o, i in (
            (shape.moe_intermediate_size, shape.hidden_size),
            (shape.moe_intermediate_size, shape.hidden_size),
            (shape.hidden_size, shape.moe_intermediate_size)))
        return b, "ANALYTIC nvfp4 estimate (weights not present)"
    b = sum(sizing.gguf_packed_bytes_analytic(o, i, 144, 256) for o, i in (
        (shape.moe_intermediate_size, shape.hidden_size),
        (shape.moe_intermediate_size, shape.hidden_size),
        (shape.hidden_size, shape.moe_intermediate_size)))
    return b, "ANALYTIC uniform-Q4_K estimate (weights not present; mixed XL runs larger)"


def pinned_bytes(base: dict, shape: ModelShape, repo_root: str,
                 tensor_parallelism: int | None = None):
    """The pinned VRAM REGION the engine reserves on every TP GPU.

    NOT the checkpoint's stored non-expert bytes.  That was the pre-2026-09
    answer and it was 41.6% low on GLM-5.3-Flash (9384.7 vs 16073.0 MiB) —
    ``TD-AUTOCONFIG-PINNED-BYTES-UPPER-BOUND``, which made the solver hand the
    TP GPU 6.7 GiB of residual that does not exist and derive an unbootable
    recipe on the default lever path.  The engine lays this region out slot by
    slot (``build_upload_plan``) and several slots are upper bounds, so the
    only faithful answer is to walk the same plan:
    ``gguf_meta.pinned_region_layout``.

    ``tensor_parallelism`` defaults to the recipe's own when it carries one and
    to 1 otherwise.  The solver SEARCHES tp after this is computed, and the
    per-rank region only shrinks as tp grows, so tp=1 is the conservative arm
    of that search — never the optimistic one.

    Fallback order after the plan walk: the checkpoint's stored bytes (an
    UNDER-estimate, labelled as one) for architectures the walk does not model,
    then the analytic parameter-count estimate when there are no weights."""
    w = (base.get("model") or {}).get("weights_path", "")
    fmt = (base.get("model") or {}).get("weights_format", "")
    tp = tensor_parallelism if tensor_parallelism else int(
        (base.get("parallelism") or {}).get("tensor_parallelism", 1) or 1)
    mem = base.get("memory") or {}
    weights_abs = ""
    for root in (repo_root, "."):
        cand = os.path.join(root, w)
        if w and os.path.exists(cand):
            weights_abs = cand
            break
    try:
        lay = gguf_meta.pinned_region_layout(
            base.get("model") or {}, base.get("quantization") or {}, tp,
            weights_abs if fmt == "gguf" else "",
            mem.get("pinned_layers"), mem.get("tp_mode_per_layer"))
        return lay.total_bytes, lay.provenance
    except (gguf_meta.UnmodelledArchitecture, KeyError, ValueError, OSError) as e:
        unmodelled = str(e)
    if fmt == "gguf" and weights_abs:
        try:
            known, unknown = gguf_meta.non_expert_bytes(weights_abs)
            return known + unknown, (
                "ESTIMATE — GGUF headers, sum of non-expert tensor STORED bytes; "
                "this UNDER-estimates the engine's pinned region (which is laid "
                "out slot by slot with upper-bound slots) because that region is "
                f"not modelled here: {unmodelled}")
        except (ValueError, OSError):
            pass
    s = shape
    per_layer_attn = (s.hidden_size * s.q_lora_rank
                      + s.q_lora_rank * s.num_attention_heads
                      * (s.qk_nope_head_dim + s.qk_rope_head_dim)
                      + s.hidden_size * (s.kv_lora_rank + s.qk_rope_head_dim)
                      + s.kv_lora_rank * s.num_attention_heads
                      * (s.qk_nope_head_dim + s.v_head_dim)
                      + s.num_attention_heads * s.v_head_dim * s.hidden_size)
    dense = 3 * s.hidden_size * s.intermediate_size
    shexp = 3 * s.hidden_size * s.moe_intermediate_size * max(s.n_shared_experts, 0)
    params = (s.num_hidden_layers * (per_layer_attn + shexp)
              + s.first_k_dense_replace * dense
              + 2 * s.vocab_size * s.hidden_size)
    return int(params * 0.6) + (2 * s.vocab_size * s.hidden_size), \
        "ANALYTIC parameter-count estimate (weights not present; +-15%, and it " \
        "models STORED parameters rather than the engine's slotted region — " \
        "direction unknown)"


def solve_recipe(base: dict, hw: HardwareDescriptor, levers: Levers,
                 knobs: AutoconfigKnobs, repo_root: str,
                 cal=None, cal_reject=(), pins=None) -> "SolveResultLike":
    """One derivation pass (AUTOCONFIG §5). Shared by the auto-run pipeline
    and the legacy ``--config`` entry point."""
    # P-29 step 13: whole-recipe form — an MTP-armed recipe (speculation.method
    # "mtp" + speculation.mtp.enabled) extends the MoE census over the
    # NextN block, which the arena/solver sizing must fund.
    shape = ModelShape.from_config(base)
    slot, slot_prov = expert_slot_bytes(base, shape, repo_root)
    # Per-RANK region, memoized per tp: the solver searches tp, and the engine's
    # own region shrinks with it (LayerRegistry divides the sharded slots).
    # Charging every lattice point the tp=1 figure would refuse fits the engine
    # takes; charging the tp=8 figure would under-reserve. Derive per tp.
    _pin_cache: dict = {}

    def pinned_for(tp: int):
        if tp not in _pin_cache:
            _pin_cache[tp] = pinned_bytes(base, shape, repo_root, tp)
        return _pin_cache[tp]

    pinned_prov = pinned_for(
        max(1, len((base.get("hardware") or {}).get("tp_array") or [1])))[1]
    pinned = lambda tp: pinned_for(tp)[0]
    draft = None
    draft_warning = ""
    ck = ((base.get("speculation") or {}).get("dspark") or {}).get("checkpoint_path", "")
    if ck:
        draft = load_draft_candidate(ck, repo_root)
        # TD-AUTOCONFIG-DRAFT-IDENTITY: a recipe-named draft is identity-
        # matched BEFORE it is ever costed — a FOREIGN draft REFUSES (it
        # once got priced at 9.14 GiB/rank and only shed by capacity
        # pressure; on a roomier box it would have booted). One that is
        # simply not on this box derives WITHOUT a draft, out loud (the
        # pre-existing outcome, no longer silent).
        if draft is None:
            draft_warning = (
                f"speculation.dspark.checkpoint_path {ck} does not exist on "
                f"this box — deriving WITHOUT a draft (the recipe asked for "
                f"one; put the checkpoint there or drop the speculation "
                f"section to silence this)")
            print("autoconfig WARNING: " + draft_warning, file=sys.stderr)
        else:
            id_errs = draft_identity_errors(draft, shape)
            if id_errs:
                raise Infeasible(
                    "draft-identity-mismatch",
                    "speculation.dspark.checkpoint_path",
                    f"{ck} as this model's speculator",
                    "a draft trained FOR this model — " + "; ".join(id_errs),
                    "a foreign draft is never priced (nor silently shed): "
                    "point at a matching .dspark checkpoint, or serve "
                    "without one (--draft none / drop the speculation "
                    "section)")
    sv = Solver(hw, shape, base, levers, knobs, cal=cal,
                cal_reject_reasons=tuple(cal_reject),
                expert_slot_bytes=slot, slot_provenance=slot_prov,
                non_expert_pinned_bytes=pinned, pinned_provenance=pinned_prov,
                draft=draft, pins=pins)
    res = sv.solve()
    if draft_warning:
        res.warnings.insert(0, draft_warning)
    return res


SolveResultLike = object  # (solver.SolveResult; annotation kept import-free)


# ------------------------------------------------------------------ options

@dataclass(frozen=True)
class AutoRunOptions:
    model_path: str
    draft_path: str = ""              # "" = discover; "none" = force no draft
    vram_expert_ratio: float | None = None
    total_active_context_tokens: int | None = None
    # levers 3/4 carry EXPLICITNESS (None = unset): a reuse of an existing
    # config is only blocked by levers the user actually passed
    # (TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS), so unset must be
    # distinguishable from an explicit default.
    prefer: str | None = None         # lever 3: E1 search ORDER (§2.3)
    accuracy: str | None = None       # lever 4: numerics FLOOR (§2.4)
    name: str = ""                    # config/artifact stem override
    prefix: str = ""                  # prepended to the default stem
    out_dir: str = ""                 # "" = cwd
    repo_root: str = "."
    reset: bool = False               # re-measure BOTH artifacts from scratch
    retrain: bool = False             # re-run training only
    skip_training: bool = False       # stop after step 2 (no GPU for step 3)
    calibration: str = ""             # use this artifact verbatim
    trained: str = ""                 # use this trained artifact verbatim
    base_config: str = ""             # advanced: carry a recipe's identity/extras
    compare_to: str = ""              # diff the result against a reference recipe
    hardware_json: str = ""           # what-if / tests instead of live detection
    tokenizer_path: str = ""          # override tokenizer discovery
    prepacked_dir: str = ""           # override prepack discovery
    placement_table: str = ""         # supply a measured M3 arena freq table
    refit_placement: bool = False     # re-fit the placement table (capturing
                                      # a fresh trace) even if one exists
    skip_placement: bool = False      # do not fit one (online placement only)
    allow_warm_holder: bool = False   # override the cold-boot guard
    pins: object | None = None        # PinSet: HARD constraints (design
                                      # spec/plans/AUTOCONFIG_PINNED_CONSTRAINTS.md)
    redetect: bool = False            # re-derive even when the config is current
    knobs: AutoconfigKnobs | None = None


@dataclass
class StepRecord:
    step: str          # calibrate | derive | train
    action: str        # measured | reused | supplied | skipped
    detail: str
    artifact: str = ""


@dataclass
class AutoRunResult:
    config_path: str
    steps: list = field(default_factory=list)
    serve_command: str = ""
    warnings: list = field(default_factory=list)
    unexplained: int = 0
    log_dir: str = ""          # where the measured steps' engine logs went


# -------------------------------------------------------------------- paths

@dataclass(frozen=True)
class ArtifactPaths:
    stem: str
    config_path: str
    explain_path: str
    compare_path: str
    autorun_path: str
    log_dir: str
    calibration_abs: str
    calibration_cfg: str      # what goes in gpu_loader.calibration_path
    trained_abs: str
    trained_cfg: str
    placement_abs: str
    placement_cfg: str        # what goes in memory.arena_placement.freq_table


def _cfg_value_for(artifact_abs: str, weights_abs: str) -> str:
    """engine.cpp:500-512 — a RELATIVE calibration_path (bare filename or
    with directories) joins the WEIGHTS dir; only an absolute path is
    verbatim. So an artifact that lives beside the weights is written as its
    bare filename, and anything else must be absolute."""
    wdir = os.path.dirname(os.path.abspath(weights_abs))
    if os.path.dirname(os.path.abspath(artifact_abs)) == wdir:
        return os.path.basename(artifact_abs)
    return os.path.abspath(artifact_abs)


def plan_paths(opts: AutoRunOptions, src: modelprobe.ModelSource) -> ArtifactPaths:
    stem = opts.name or (opts.prefix + src.display_name)
    out_dir = os.path.abspath(opts.out_dir or os.getcwd())
    wdir = os.path.dirname(os.path.abspath(src.weights_abs))
    # Calibration lives beside the weights (the engine's own convention:
    # it is weight/config-specific). A read-only model directory falls back
    # to the output directory, and then the config carries an absolute path.
    home = wdir if os.access(wdir, os.W_OK) else out_dir
    cal_abs = os.path.abspath(opts.calibration) if opts.calibration else \
        os.path.join(home, f"gpu_loader_calibration_{stem}.json")
    if opts.trained:
        trained_abs = os.path.abspath(opts.trained)
    else:
        root, ext = os.path.splitext(cal_abs)
        trained_abs = root + "_trained" + ext
        if opts.calibration and not os.access(os.path.dirname(trained_abs), os.W_OK):
            trained_abs = os.path.join(
                home, os.path.basename(root) + "_trained" + ext)
    # The placement table resolves against the ENGINE's working directory
    # (schema: same as LS_ARENA_PLACE_FREQ) and is FAIL-CLOSED — an
    # unreadable configured table aborts engine init. A generated one is
    # therefore written as an absolute path; a user-supplied string is kept
    # verbatim (it is theirs, and usually repo-relative like the champion's).
    place_abs = os.path.abspath(opts.placement_table) if opts.placement_table \
        else os.path.join(home, f"arena_placement_{stem}.csv")
    place_cfg = opts.placement_table if opts.placement_table else place_abs
    config_path = os.path.join(out_dir, stem + ".autoconfig.json")
    return ArtifactPaths(
        stem=stem,
        config_path=config_path,
        explain_path=config_path[: -len(".json")] + ".explain.md",
        compare_path=config_path[: -len(".json")] + ".compare.md",
        autorun_path=config_path[: -len(".json")] + ".autorun.json",
        log_dir=os.path.join(out_dir, stem + ".autorun-logs"),
        calibration_abs=cal_abs,
        calibration_cfg=_cfg_value_for(cal_abs, src.weights_abs),
        trained_abs=trained_abs,
        trained_cfg=_cfg_value_for(trained_abs, src.weights_abs),
        placement_abs=place_abs,
        placement_cfg=place_cfg,
    )


# ------------------------------------------------------------- base recipe

def base_from_source(src: modelprobe.ModelSource, draft_path: str,
                     calibration_cfg: str, base_config: dict | None,
                     placement_table: str = "") -> dict:
    """The identity half of a recipe: what the solver must be TOLD (which
    model, which weights, which draft), as opposed to what it derives. With
    --model this is synthesised from the weights themselves."""
    base: dict = dict(base_config or {})
    base["model"] = dict(src.model_section)
    base["quantization"] = dict(src.quantization)
    if src.prepacked_dir:
        base["preprocessing"] = {"prepacked_dir": src.prepacked_dir}
    elif src.live_prepack:
        base["preprocessing"] = {"live_prepack": True}
    serving = dict(base.get("serving") or {})
    serving.setdefault("host", "0.0.0.0")
    serving.setdefault("port", 8000)
    serving["tokenizer_path"] = src.tokenizer_path
    base["serving"] = serving
    if draft_path and draft_path != "none":
        spec = dict(base.get("speculation") or {})
        spec["dspark"] = dict(spec.get("dspark") or {},
                              checkpoint_path=draft_path)
        base["speculation"] = spec
    elif draft_path == "none":
        base.pop("speculation", None)
    if placement_table:
        mem = dict(base.get("memory") or {})
        mem["arena_placement"] = {"freq_table": placement_table}
        base["memory"] = mem
    base["gpu_loader"] = {"enabled": True, "calibration_mode": "loaded",
                          "calibration_path": calibration_cfg}
    return base


def _load_accept(cal_abs: str, hw: HardwareDescriptor, shape: ModelShape):
    """Load an artifact and mirror the engine's accept predicate
    (engine.cpp:519-568 / INV-LOADER-CAL-6)."""
    cal = load_calibration(cal_abs)
    order = [g.ordinal for g in sorted(hw.gpus, key=lambda g: (-g.vram_mib, g.ordinal))]
    rej = accept_calibration(cal, hw, order, shape.hidden_size,
                             shape.moe_intermediate_size)
    return cal, rej


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def serve_command(repo_root: str, config_path: str) -> str:
    """The one line the user runs afterwards, as many times as they like.

    Copy-pasteable from anywhere: the cd is part of it, because the serve
    stack's PYTHONPATH (build/python + python) is repo-root-relative.
    CUDA_DEVICE_ORDER=PCI_BUS_ID is part of it too
    (TD-AUTOCONFIG-SERVE-CMD-DEVICE-ORDER): the recipe's ``gpus[].id`` are
    PCI-ordered (hwdetect's contract; enginerun.base_env() sets the same
    for the pipeline's own boots) — without it a default fastest-first
    ordering lands a 5090-sized carve on a 5080 and the boot dies with
    ``device_alloc failed``."""
    root = os.path.abspath(repo_root)
    py = os.path.join(root, ".venv", "bin", "python")
    py = os.path.relpath(py, root) if os.path.exists(py) else sys.executable
    cfg = os.path.relpath(config_path, root) \
        if os.path.abspath(config_path).startswith(root + os.sep) else config_path
    return (f"cd {root} && CUDA_DEVICE_ORDER=PCI_BUS_ID "
            f"PYTHONPATH=build/python:python {py} "
            f"python/cli/serve.py --config {cfg}")


# ----------------------------------------------------------------- the run


def _current_config(paths: ArtifactPaths, hw: HardwareDescriptor,
                    shape: ModelShape, opts: AutoRunOptions):
    """``(fingerprint, "")`` when an existing config is still valid for this
    box, every artifact it names is present AND it matches every lever the
    user explicitly passed; ``("", reason)`` when only an explicit lever
    blocks the reuse (TD-AUTOCONFIG-LEGACY-CONFIG-MODE-IGNORES-LEVERS: a
    passed lever must change the output or refuse — never be silently
    discarded by a cache that does not include it); ``("", "")`` otherwise."""
    if not os.path.exists(paths.config_path):
        return "", ""
    try:
        with open(paths.config_path) as f:
            existing = json.load(f)
    except (OSError, ValueError):
        return "", ""
    stored = (existing.get("autoconfig") or {}).get("fingerprint", "")
    if not stored:
        return "", ""
    named = (existing.get("gpu_loader") or {}).get("calibration_path", "")
    named_abs = resolve_calibration_path(named, os.path.join(
        os.path.abspath(opts.repo_root),
        (existing.get("model") or {}).get("weights_path", "")))
    if not os.path.exists(named_abs):
        return "", ""
    try:
        cal, rej = _load_accept(named_abs, hw, shape)
    except (OSError, ValueError):
        return "", ""
    if rej:
        return "", ""
    table = ((existing.get("memory") or {}).get("arena_placement") or {}).get(
        "freq_table", "")
    if table:
        t_abs = table if os.path.isabs(table) else os.path.join(
            os.path.abspath(opts.repo_root), table)
        if not os.path.exists(t_abs):
            return "", ""    # fail-closed at boot; re-derive instead
    elif not opts.skip_placement:
        return "", ""        # unfinished: step 4 never produced a table
    # the fingerprint folds the calibration's tier matrix, so compare against
    # the artifact the config itself names
    if fingerprint(hw, cal) != stored:
        return "", ""
    mism = _explicit_lever_mismatches(existing, opts)
    if mism:
        return "", "; ".join(mism)
    return stored, ""


def _explicit_lever_mismatches(existing: dict, opts: AutoRunOptions) -> list:
    """Explicitly passed levers the stored config was NOT derived with.
    Unset levers (None) never block a reuse — 'tuned once, not every run'
    survives; an explicit ask must never be silently discarded."""
    ac = existing.get("autoconfig") or {}
    mism = []
    for name, req, stored in (
            ("vram_expert_ratio", opts.vram_expert_ratio,
             ac.get("vram_expert_ratio")),
            ("total_active_context_tokens", opts.total_active_context_tokens,
             ac.get("total_active_context_tokens")),
            ("prefer", opts.prefer, ac.get("prefer") or "balanced"),
            ("accuracy", opts.accuracy, ac.get("accuracy") or "standard")):
        if req is not None and req != stored:
            mism.append(f"{name}: stored {stored!r} vs requested {req!r}")
    return mism


def run_auto(opts: AutoRunOptions, log=print) -> AutoRunResult:
    """Steps 1-3 end to end. Raises Infeasible/EngineRunError on refusal."""
    repo_root = os.path.abspath(opts.repo_root)
    knobs = opts.knobs or AutoconfigKnobs()
    # Supplying an artifact and asking to re-measure it are contradictory
    # asks; overwriting a file the user pointed at would be the worst reading.
    for supplied, flag in ((opts.calibration, "--reset" if opts.reset else ""),
                           (opts.trained, "--reset" if opts.reset else
                            "--retrain" if opts.retrain else ""),
                           (opts.placement_table, "--reset" if opts.reset else
                            "--refit-placement" if opts.refit_placement else "")):
        if supplied and flag:
            raise Infeasible(
                "conflicting-flags", flag, f"{flag} with a supplied artifact",
                "either a supplied artifact or a fresh measurement",
                f"drop {flag} to use {os.path.basename(supplied)}, or drop "
                f"the --calibration/--trained path to re-measure")
    base_config = None
    if opts.base_config:
        with open(opts.base_config) as f:
            base_config = json.load(f)

    # step 0 — what model is this?
    src = modelprobe.probe_model(
        opts.model_path, repo_root,
        prepacked=opts.prepacked_dir, tokenizer=opts.tokenizer_path,
        base_model_section=(base_config or {}).get("model"),
        base_quantization=(base_config or {}).get("quantization"))
    log(f"autoconfig: model {src.display_name} — {src.model_section['architecture']}, "
        f"{src.model_section.get('num_hidden_layers')} layers, "
        f"{src.model_section.get('n_routed_experts')} experts "
        f"[{src.source}]")
    for line in src.provenance:
        log(f"autoconfig:   {line}")

    shape = ModelShape.from_model_section(
        src.model_section,
        # P-29 step 13: the probed model section carries no speculation —
        # the arming signal comes from the supplied base recipe (the
        # auto-run derives its own speculation later; solve_recipe
        # re-derives the shape from the recipe it is about to solve).
        mtp_experts_armed=mtp_experts_armed(base_config or {}))

    # Draft resolution (TD-AUTOCONFIG-DRAFT-IDENTITY): identity is matched
    # BEFORE anything is costed or booted. An explicit --draft that is
    # missing or foreign REFUSES; discovery SKIPS foreign candidates out
    # loud and never wires one in.
    draft = opts.draft_path
    if draft and draft != "none":
        dc = load_draft_candidate(draft, repo_root)
        if dc is None:
            raise Infeasible(
                "draft-checkpoint-missing", "--draft", draft,
                "an existing .dspark checkpoint directory",
                "check --draft, or pass --draft none to serve without one")
        id_errs = draft_identity_errors(dc, shape)
        if id_errs:
            raise Infeasible(
                "draft-identity-mismatch", "--draft",
                f"{draft} as {src.display_name}'s speculator",
                "a draft trained FOR this model — " + "; ".join(id_errs),
                "a foreign draft is never priced (nor silently shed): point "
                "--draft at a matching .dspark checkpoint, or pass "
                "--draft none")
    elif not draft:
        for cand in modelprobe.find_draft_checkpoints(src.weights_abs,
                                                      repo_root):
            dc = load_draft_candidate(cand, repo_root)
            id_errs = draft_identity_errors(dc, shape) if dc is not None \
                else ["unreadable checkpoint (no config.json)"]
            if not id_errs:
                draft = cand
                log(f"autoconfig:   speculator {draft} (discovered, "
                    f"identity-matched; --draft none to serve without one)")
                break
            log(f"autoconfig:   speculator candidate {cand} REJECTED — not "
                f"this model's draft: {'; '.join(id_errs)} (a foreign draft "
                f"is never priced)")

    paths = plan_paths(opts, src)
    hw = detect_hardware() if not opts.hardware_json else \
        HardwareDescriptor.from_json(json.load(open(opts.hardware_json)))
    if not hw.gpus:
        raise Infeasible("no-gpus-detected", "box", "at least one NVIDIA GPU",
                         "none found in /proc/driver/nvidia/gpus",
                         "autoconfig fits GPU serving; nothing to derive here")
    levers = Levers(opts.vram_expert_ratio, opts.total_active_context_tokens,
                    opts.prefer or "balanced", opts.accuracy or "standard")

    # Tuned once, not every run (AUTOCONFIG §1): an existing config whose
    # fingerprint matches this box, and whose artifacts are all present, is
    # reused VERBATIM — hand edits survive. --reset/--redetect re-derive.
    if not (opts.reset or opts.redetect or opts.retrain or opts.pins):
        # pins force a re-derivation: reusing a stored config would
        # silently ignore a HARD constraint (design §2 corollary)
        current, lever_mism = _current_config(paths, hw, shape, opts)
        if lever_mism:
            log(f"autoconfig: {os.path.basename(paths.config_path)} matches "
                f"this hardware but NOT the requested levers ({lever_mism}) "
                f"— re-deriving (an explicit lever is never silently "
                f"discarded)")
        elif current:
            log(f"autoconfig: {os.path.basename(paths.config_path)} is already "
                f"current for this box ({current}) — reusing it verbatim "
                f"(--redetect to re-derive, --reset to re-measure)")
            res = AutoRunResult(config_path=paths.config_path)
            res.steps = [StepRecord("calibrate", "reused", "config is current",
                                    paths.calibration_abs),
                         StepRecord("derive", "reused", "fingerprint matches",
                                    paths.config_path),
                         StepRecord("train", "reused", "config is current",
                                    paths.trained_abs),
                         StepRecord("place", "reused", "config is current",
                                    paths.placement_abs)]
            res.serve_command = serve_command(repo_root, paths.config_path)
            res.log_dir = paths.log_dir
            log("")
            log("autoconfig: DONE — serve this model with:")
            log("")
            log("    " + res.serve_command)
            return res
    env = EngineEnv(repo_root=repo_root)
    steps: list = []
    result = AutoRunResult(config_path=paths.config_path)

    # A table that already exists (or was supplied) is carried INTO the
    # derivation, so the solver explains it as a carried measurement rather
    # than as an absence. A freshly fitted one is added after step 4.
    placement_in = ""
    if opts.placement_table:
        if not os.path.exists(paths.placement_abs):
            raise Infeasible("supplied-placement-missing", opts.placement_table,
                             opts.placement_table, "an existing freq table",
                             "check --placement-table, or drop it to fit one")
        placement_in = paths.placement_cfg
    elif os.path.exists(paths.placement_abs) and not (opts.reset
                                                      or opts.refit_placement):
        placement_in = paths.placement_cfg

    # ---- step 1: CALIBRATE ------------------------------------------------
    cal, rej = None, ()
    have = os.path.exists(paths.calibration_abs)
    if opts.calibration and not have:
        raise Infeasible("supplied-calibration-missing", opts.calibration,
                         opts.calibration, "an existing artifact",
                         "check --calibration, or drop it to measure one")
    if have and not opts.reset:
        cal, rej = _load_accept(paths.calibration_abs, hw, shape)
        if rej:
            log("autoconfig: step 1/4 calibrate — artifact REJECTED: "
                + "; ".join(rej))
            if opts.calibration:
                raise Infeasible(
                    "supplied-calibration-rejected", opts.calibration,
                    "; ".join(rej), "an artifact measured on THIS box for "
                    "THIS model", "drop --calibration to measure a new one")
            cal = None
        else:
            action = "supplied" if opts.calibration else "reused"
            steps.append(StepRecord("calibrate", action,
                                    "matches this box and model "
                                    "(UUIDs + compute dims)",
                                    paths.calibration_abs))
            log(f"autoconfig: step 1/4 calibrate — {action} "
                f"{os.path.basename(paths.calibration_abs)}")
    if cal is None:
        # Measure: emit a PROVISIONAL recipe (calibration_mode full) and boot
        # it once. The engine is the only thing that can measure this.
        base = base_from_source(src, draft, paths.calibration_cfg, base_config,
                                placement_in)
        prov = solve_recipe(base, hw, levers, knobs, repo_root,
                            pins=opts.pins)
        prov.recipe["gpu_loader"] = {"enabled": True,
                                     "calibration_mode": "full",
                                     "calibration_path": paths.calibration_cfg}
        prov_path = paths.config_path[: -len(".json")] + ".provisional.json"
        _write(prov_path, json.dumps(prov.recipe, indent=2) + "\n")
        log("autoconfig: step 1/4 calibrate — measuring on this box "
            f"(engine boot, ~5-10 min cold; log {paths.log_dir}/calibrate.log)")
        run_calibration_boot(
            prov_path, env, os.path.join(paths.log_dir, "calibrate.log"),
            paths.calibration_abs, tokenizer_path=(
                src.tokenizer_path if src.tokenizer_path != "auto" else ""),
            allow_warm_holder=opts.allow_warm_holder)
        cal, rej = _load_accept(paths.calibration_abs, hw, shape)
        if rej:
            raise EngineRunError(
                "the engine wrote a calibration this box does not accept: "
                + "; ".join(rej))
        steps.append(StepRecord("calibrate", "measured",
                                "engine full calibration on this box",
                                paths.calibration_abs))
        os.remove(prov_path)

    # ---- step 2: DERIVE ---------------------------------------------------
    base = base_from_source(src, draft, paths.calibration_cfg, base_config,
                            placement_in)
    solved = solve_recipe(base, hw, levers, knobs, repo_root, cal=cal,
                          cal_reject=rej, pins=opts.pins)
    recipe = solved.recipe
    fp = fingerprint(hw, cal)
    recipe["autoconfig"] = {
        "enabled": True,
        "vram_expert_ratio": levers.vram_expert_ratio,
        "total_active_context_tokens": levers.total_active_context_tokens,
        "output_path": paths.config_path,
        "fingerprint": fp,
    }
    # Schema-backed levers: carried ONLY when non-default (see cli.py) —
    # default derivations stay byte-identical, non-default choices survive
    # a re-derivation from the emitted recipe alone.
    if levers.prefer != "balanced":
        recipe["autoconfig"]["prefer"] = levers.prefer
    if levers.accuracy != "standard":
        recipe["autoconfig"]["accuracy"] = levers.accuracy
    solved.explanations.add(
        "autoconfig.fingerprint", fp, "measured",
        "measured-topology fingerprint (GPU UUIDs/NUMA/PCIe/VRAM + the "
        "calibration's tier matrix); re-derivation only when it changes")
    _write(paths.config_path, json.dumps(recipe, indent=2) + "\n")
    steps.append(StepRecord("derive", "measured",
                            f"{len(solved.explanations.all())} explained "
                            f"fields; {len(solved.degradations)} degradation(s)",
                            paths.config_path))
    log(f"autoconfig: step 2/4 derive — wrote {os.path.basename(paths.config_path)}")
    result.warnings = list(solved.warnings)

    # ---- step 3: TRAIN ----------------------------------------------------
    trained_ok = False
    train_action = ""
    if opts.trained and not os.path.exists(paths.trained_abs):
        raise Infeasible("supplied-trained-missing", opts.trained, opts.trained,
                         "an existing trained artifact",
                         "check --trained, or drop it to train one")
    reuse_trained = (os.path.exists(paths.trained_abs)
                     and not opts.reset and not opts.retrain)
    if reuse_trained:
        _, trej = _load_accept(paths.trained_abs, hw, shape)
        if trej and not opts.trained:
            log("autoconfig: step 3/4 train — existing trained artifact "
                "REJECTED (" + "; ".join(trej) + "); retraining")
            reuse_trained = False
        elif trej:
            raise Infeasible(
                "supplied-trained-rejected", opts.trained, "; ".join(trej),
                "an artifact measured on THIS box for THIS model",
                "drop --trained to train a new one")
    if reuse_trained:
        action = train_action = "supplied" if opts.trained else "reused"
        steps.append(StepRecord("train", action,
                                "workload-corrected constants already present",
                                paths.trained_abs))
        log(f"autoconfig: step 3/4 train — {action} "
            f"{os.path.basename(paths.trained_abs)}")
        trained_ok = True
    elif opts.skip_training:
        train_action = "skipped"
        steps.append(StepRecord("train", "skipped",
                                "--skip-training: the recipe serves on the "
                                "hardware calibration alone"))
        log("autoconfig: step 3/4 train — SKIPPED (--skip-training); the "
            "recipe uses the uncorrected hardware calibration")
    else:
        log("autoconfig: step 3/4 train — measuring a real 100-token decode "
            f"and fitting the constants to it (log {paths.log_dir}/train.log)")
        run_training_boot(
            paths.config_path, env, os.path.join(paths.log_dir, "train.log"),
            os.path.join(paths.log_dir, "loader_shadow.jsonl"),
            os.path.join(paths.log_dir, "perf_trace.csv"),
            tokenizer_path=(src.tokenizer_path
                            if src.tokenizer_path != "auto" else ""))
        run_trainer_apply(
            repo_root, paths.calibration_abs,
            os.path.join(paths.log_dir, "loader_shadow.jsonl"),
            os.path.join(paths.log_dir, "perf_trace.csv"),
            paths.trained_abs, os.path.join(paths.log_dir, "trainer.log"),
            python=env.interpreter())
        train_action = "measured"
        steps.append(StepRecord("train", "measured",
                                "100-token decode fitted into corrected "
                                "loader constants", paths.trained_abs))
        trained_ok = True

    # The trained artifact is the one the engine loads from now on: it is
    # the second, better iteration of the same constants.
    if trained_ok:
        recipe["gpu_loader"] = {"enabled": True, "calibration_mode": "loaded",
                                "calibration_path": paths.trained_cfg}
        solved.explanations.add(
            "gpu_loader.calibration_path", paths.trained_cfg, "measured",
            "the TRAINED constants (hardware calibration corrected against a "
            "real 100-token decode) — step 3 of the auto-run; the untrained "
            f"baseline stays at {os.path.basename(paths.calibration_abs)}",
            refs=("spec/AUTO_RUN.md", "tools/loader_xray/trainer_apply.py"))
        _write(paths.config_path, json.dumps(recipe, indent=2) + "\n")

    # ---- step 4: PLACE ----------------------------------------------------
    # The host arena places expert slots across NUMA banks by per-(layer,
    # expert) DEMAND-FETCH frequency. That table is measured, not derivable.
    # When step 3 ran, its decode already produced the trace this fits from
    # and step 4 costs no boot; otherwise step 4 captures a trace of its own.
    trace = os.path.join(paths.log_dir, "perf_trace.csv")
    fitted = False
    if placement_in:
        action = "supplied" if opts.placement_table else "reused"
        steps.append(StepRecord("place", action,
                                "measured demand-fetch table already present",
                                paths.placement_abs))
        log(f"autoconfig: step 4/4 place — {action} "
            f"{os.path.basename(paths.placement_abs)}")
    elif opts.skip_placement:
        steps.append(StepRecord("place", "skipped",
                                "--skip-placement: the online migrator "
                                "converges the layout on its own"))
        log("autoconfig: step 4/4 place — SKIPPED (--skip-placement)")
    else:
        have_trace = os.path.exists(trace) and os.path.getsize(trace) > 0
        if opts.refit_placement and train_action != "measured":
            have_trace = False   # an explicit re-fit wants a FRESH trace
        # THE PIPELINE FINISHES (user directive, 2026-08-30). Supplying or
        # reusing an artifact means "do not redo THAT step" — it never means
        # "stop short". So a missing placement table is unfinished work no
        # matter how steps 1 and 3 were satisfied, and step 4 captures the
        # trace it needs by itself. Whatever satisfied step 3 is left
        # UNTOUCHED: the capture boot writes a perf trace and nothing else
        # (no shadow dump, no trainer_apply). --skip-placement is the one way
        # to end up without a table.
        if not have_trace:
            if train_action in ("supplied", "reused"):
                log("autoconfig: step 4/4 place — the trained calibration "
                    f"{os.path.basename(paths.trained_abs)} is REUSED, not "
                    "re-fitted; this boot only captures the decode trace the "
                    "placement fit needs.")
            log("autoconfig: step 4/4 place — capturing a decode trace "
                f"(engine boot + 100 tokens; log "
                f"{paths.log_dir}/place_capture.log)")
            run_capture_boot(
                paths.config_path, env,
                os.path.join(paths.log_dir, "place_capture.log"), trace,
                tokenizer_path=(src.tokenizer_path
                                if src.tokenizer_path != "auto" else ""))
        run_freq_table_fit(repo_root, paths.placement_abs, [trace],
                           os.path.join(paths.log_dir, "freq_table.log"),
                           python=env.interpreter())
        fitted = True
        detail = ("demand-fetch table fitted from the 100-token decode trace"
                  if train_action == "measured" else
                  "demand-fetch table fitted from a captured decode trace "
                  "(the trained calibration was left untouched)")
        steps.append(StepRecord("place", "measured", detail,
                                paths.placement_abs))
        log(f"autoconfig: step 4/4 place — fitted "
            f"{os.path.basename(paths.placement_abs)} from the decode trace")

    if fitted:
        mem = recipe.setdefault("memory", {})
        mem["arena_placement"] = {"freq_table": paths.placement_cfg}
        solved.explanations.add(
            "memory.arena_placement", mem["arena_placement"], "measured",
            "per-(layer,expert) demand-fetch table fitted from step 3's own "
            "decode trace (tools/loader_xray/freq_table.py counts dispatched "
            "expert H2D copies, excluding prefill and warm-up sweeps). It is "
            "a BOOTSTRAP from one regime: accumulate more traces into the "
            "same table when the serving mix widens, and the M3b online "
            "migrator refines it live either way. NOTE the table's content "
            "hash folds into the ArenaCache store identity — the first boot "
            "with it rebuilds the warm store ONCE",
            refs=("arena_placement.h", "RUN.md 'M3 arena placement table'",
                  "spec/AUTO_RUN.md step 4"),
            measurement=f"fitted from {os.path.basename(trace)}")
        _write(paths.config_path, json.dumps(recipe, indent=2) + "\n")

    # ---- sidecars + report ------------------------------------------------
    explain = solved.explanations.render_markdown(
        f"Autoconfig derivation — {os.path.basename(paths.config_path)}")
    extra = ["", "## Auto-run steps", "",
             "| step | action | detail | artifact |", "|---|---|---|---|"]
    for s in steps:
        extra.append(f"| {s.step} | {s.action} | {s.detail} | "
                     f"`{os.path.basename(s.artifact) if s.artifact else ''}` |")
    if solved.warnings:
        extra += ["", "## Warnings", ""] + [f"- {w}" for w in solved.warnings]
    if solved.degradations:
        extra += ["", "## Degradations (E1)", ""] + \
            [f"- {d}" for d in solved.degradations]
    _write(paths.explain_path, explain + "\n".join(extra) + "\n")

    manifest = {
        "model": {"path": src.weights_path, "architecture":
                  src.model_section["architecture"], "source": src.source,
                  "display_name": src.display_name},
        "stem": paths.stem,
        "levers": {"vram_expert_ratio": levers.vram_expert_ratio,
                   "total_active_context_tokens":
                       levers.total_active_context_tokens,
                   "prefer": levers.prefer,
                   "accuracy": levers.accuracy},
        "fingerprint": fp,
        "artifacts": {"config": paths.config_path,
                      "calibration": paths.calibration_abs,
                      "trained": paths.trained_abs if trained_ok else "",
                      "placement": paths.placement_abs
                      if (fitted or placement_in) else "",
                      "explain": paths.explain_path},
        "steps": [s.__dict__ for s in steps],
        "warnings": solved.warnings,
    }
    if opts.pins:
        # only when present: default artifacts stay byte-identical, and the
        # manifest records what the derivation was TOLD (not derived)
        manifest["pins"] = {path: pin.value
                            for path, pin in opts.pins.items()}
    _write(paths.autorun_path, json.dumps(manifest, indent=2) + "\n")

    if opts.compare_to:
        with open(opts.compare_to) as f:
            ref = json.load(f)
        divs = compare(ref, recipe, solved.explanations,
                       template=templates.template_view(
                           (recipe.get("model") or {})
                           .get("architecture", "")))
        _write(paths.compare_path,
               render_report(os.path.basename(opts.compare_to), divs))
        result.unexplained = sum(1 for d in divs if d.klass == "unexplained")
        log(f"autoconfig: compared against {os.path.basename(opts.compare_to)} "
            f"— {summary_line(divs)} Only unexplained gates acceptance "
            f"({os.path.basename(paths.compare_path)})")

    result.steps = steps
    result.log_dir = paths.log_dir
    result.serve_command = serve_command(repo_root, paths.config_path)
    for w in solved.warnings:
        log(f"autoconfig WARNING: {w}")
    log("")
    log("autoconfig: DONE — serve this model with:")
    log("")
    log("    " + result.serve_command)
    log("")
    log(f"autoconfig: (config {paths.config_path}; "
        f"why: {os.path.basename(paths.explain_path)})")
    return result


# --------------------------------------------------------- benchmark mode

def benchmark_paths(config_path: str, out_path: str = "",
                    log_dir: str = "") -> tuple:
    """Where a benchmark's artifacts go: beside the recipe it measured,
    named after it, so several arms' results sit next to their configs and
    concatenate cleanly.

    THE ENGINE LOG FOLLOWS THE RESULTS, NOT THE RECIPE.  Two runs of the SAME
    recipe with different ``--benchmark-out`` (a ladder split into a cheap
    boot and an expensive one, or an arm re-measured after a fix) used to
    share one ``<recipe>.benchmark-logs/`` and SILENTLY OVERWRITE each other,
    so every ledger row citing that path pointed at whichever run went last —
    the evidence for a number was the log of a different number
    (TD-AUTOCONFIG-BENCHMARK-LOG-DIR-COLLIDES, caught on the GF3 ladder where
    arm ctx100k's cited log actually held the ctx100k-low top-up).  So the log
    dir is derived from the OUT stem when one is given; an explicit
    ``log_dir`` still wins over both.
    """
    def _stem(path: str) -> str:
        return path[: -len(".json")] if path.endswith(".json") else path

    out = os.path.abspath(out_path or (_stem(config_path) + ".benchmark.json"))
    return (out,
            os.path.abspath(log_dir or (_stem(out) + "-logs")))


def run_benchmark_mode(config_path: str, repo_root: str,
                       opts: "BenchmarkOptions", log_dir: str = "",
                       log=print) -> dict:
    """AUTO_RUN benchmark mode (spec/AUTO_RUN.md 'benchmark'): boot the
    recipe ONCE, measure the prefill ladder and generation tok/s, write the
    machine-readable results, and EXIT.

    It is a MODE, not a fifth pipeline step, for one reason: steps 1-4
    produce the recipe, and every one of them is skippable once its artifact
    exists.  A benchmark produces no artifact the engine consumes and is
    never reusable — its whole value is that it was measured NOW, on THIS
    box, in THIS state.  Folding it into the pipeline would put ~an hour of
    engine time on the path of every `autoconfigure --model` run and would
    make `--reset` mean two different things.  So: same contract (boot,
    measure, stop, print, exit), separate switch.
    """
    if not os.path.exists(config_path):
        raise Infeasible("benchmark-config-missing", config_path, config_path,
                         "an existing recipe to measure",
                         "check the path, or derive one first with "
                         "--model <weights>")
    root = os.path.abspath(repo_root)
    out_json, logs = benchmark_paths(config_path, opts.out_path, log_dir)
    env = EngineEnv(repo_root=root)
    log(f"autoconfig: benchmark — measuring {os.path.basename(config_path)} "
        f"(one boot; logs {logs})")
    record = run_benchmark(config_path, env, logs, opts, log=log)
    write_artifacts(record, out_json, log=log)
    summarize(record, log=log)
    log("")
    log("autoconfig: benchmark DONE — no engine is left running. Serve this "
        "recipe when you want to:")
    log("")
    log("    " + serve_command(root, config_path))
    return record


# ---------------------------------------------- serve.py --model entry (P-31)

def derive_for_serve(model_path: str, *, repo_root: str = ".",
                     base_config: str = "", out_dir: str = "",
                     vram_expert_ratio: float | None = None,
                     total_active_context_tokens: int | None = None,
                     prefer: str | None = None, accuracy: str | None = None,
                     pins=None, redetect: bool = False) -> tuple[int, str]:
    """CPU-only derive for ``serve.py --autoconfig --model <weights>``
    (TD-AUTOCONFIG-NO-GLM5NEXT-PROFILE resolution's consumer): probe the
    weights (AUTO_RUN step 0), reuse the measured artifacts beside them
    (trained loader calibration, arena placement table), derive (step 2)
    and return ``(exit_code, recipe_path)``.  NEVER boots an engine — a
    missing calibration derives ``calibration_mode: full`` and the serving
    boot self-heals (engine.cpp:519-568), it does not run the auto-run
    pipeline's measured steps.  Serve then boots the returned recipe."""
    from .cli import run as _cli_run
    base_cfg: dict = {}
    if base_config:
        with open(base_config) as f:
            base_cfg = json.load(f)
    src = modelprobe.probe_model(
        model_path, repo_root,
        base_model_section=(base_cfg.get("model") or None),
        base_quantization=(base_cfg.get("quantization") or None))
    opts = AutoRunOptions(model_path=model_path, repo_root=repo_root,
                          out_dir=out_dir)
    paths = plan_paths(opts, src)
    # discovery SKIPS foreign candidates out loud, exactly like run_auto
    # (TD-AUTOCONFIG-DRAFT-IDENTITY): a .dspark merely lying near the
    # weights proves nothing about WHOSE speculator it is
    shape = ModelShape.from_model_section(src.model_section)
    draft = ""
    for cand in modelprobe.find_draft_checkpoints(src.weights_abs, repo_root):
        dc = load_draft_candidate(cand, repo_root)
        id_errs = draft_identity_errors(dc, shape) if dc is not None \
            else ["unreadable checkpoint (no config.json)"]
        if not id_errs:
            draft = cand
            break
        print(f"autoconfig: speculator candidate {cand} REJECTED — not "
              f"this model's draft: {'; '.join(id_errs)} (a foreign draft "
              "is never priced)", file=sys.stderr)
    # prefer the TRAINED calibration when it exists (auto-run step 3's
    # repoint, done here for the derive-only path); the untrained one and
    # a bare 'full' mode are the self-healing fallbacks, in that order
    if os.path.exists(paths.trained_abs):
        cal_cfg = paths.trained_cfg
    else:
        cal_cfg = paths.calibration_cfg
    placement_in = paths.placement_cfg \
        if os.path.exists(paths.placement_abs) else ""
    base = base_from_source(src, draft, cal_cfg, base_cfg, placement_in)
    rc = _cli_run("", paths.config_path,
                  vram_expert_ratio=vram_expert_ratio,
                  total_active_context_tokens=total_active_context_tokens,
                  redetect=redetect, repo_root=repo_root,
                  prefer=prefer, accuracy=accuracy, pins=pins,
                  base_override=base)
    return rc, paths.config_path
