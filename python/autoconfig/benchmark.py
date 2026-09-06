"""AUTO_RUN benchmark mode — measure a derived recipe, then EXIT.

    python python/cli/autoconfigure.py --benchmark --config <recipe>.json
    python python/cli/autoconfigure.py --model <weights> --benchmark

One engine boot, two measurements, machine-readable results, no engine left
running:

  1. PREFILL LADDER   served prefill tok/s at several prompt lengths, scaled
                      to the recipe's OWN ``serving.max_sequence_length``
                      (the point is to benchmark long-context recipes, so a
                      hardcoded 8k/20k/25k ladder would be a lie on a 1M
                      config).  Every point is REPEATED — the campaign's 20k
                      point carries a documented +-1.3 % variance class
                      (spec/measurements/glm_prefill.md), so a single number
                      per point is not a measurement.
  2. GENERATION       decode tok/s, B=1, greedy, n runs (GF3.15's shape:
                      5 x 300 tokens).  The median AND every individual value
                      are reported, plus whether the runs were
                      output-identical — that identity is itself a
                      correctness signal (spec/reports/GF315_ECONOMICS.md).

WHERE THE NUMBERS COME FROM.  Both are the SERVER's own per-request
accounting, the same ``[orch-stats]`` line the serving-gap ledgers quote:
``prefill_ms`` (prefill wall, queueing excluded) and ``decode_tok_s``
(committed tokens / decode wall, prefill excluded — RequestStats.tok_per_s).
Nothing here re-derives a rate from a client-side stopwatch.

WHAT EVERY SAMPLE CARRIES (dossier 4b, MANDATORY).  A tok/s number without
its preconditions is not evidence:

  * ``deterministic_ep_combine`` state (resolved from the recipe + the env
    the runner sets, and cross-checked against the engine's own boot notice);
  * the DET-TOPK-TIES kernel's presence (a BUILD property — dossier 4b says
    the runner must verify it; ``deps/LayerStoRmKernels`` >= 8228e71);
  * ``moe_degraded_layers`` per request.  Nonzero means that answer was
    computed with an INCOMPLETE expert set: the sample is DISCARDED from
    every aggregate and the discard is stated in the output and on stdout.
  * ``indexer_dense_steps`` per request (TD-INDEXER-NO-DENSE-FALLBACK: a
    nonzero count is a ~10x-slower bug witness, not load) — also discarded.
  * the GENERATED TEXT itself (TD-GLM53-EP4-DEGENERATE-GENERATION).  The
    counters above answer "was anything LATE or DEAD", never "is the answer
    a sentence".  A shape that computes a quarter of its experts produces
    fluent-looking degenerate text with both counters at zero — and
    repetitive text routes to a repetitive expert set, which INFLATES the
    decode rate, so the counter-clean run reports a number that is better
    than the truth.  Every sample therefore carries ``repetition_ratio``
    (distinct 4-grams / 4-grams of the completion) and ``prompt_echo_ratio``
    (completion 4-grams that appear verbatim in the prompt), and a sample
    that LOOPS or ECHOES hard is discarded like a degraded one.

WARM vs COLD.  The arena's warmth dominates boot wall and the first
requests' fetch behaviour; a chart that mixes a warm-holder boot with a cold
store rebuild is a lie.  So the record carries the holder pids seen before
the boot, the host ``Shmem`` before and after, the boot wall, and the
engine's own ``arena_attach``/``live prepack`` lines.

NEVER TWO ENGINES (dossier 1).  The benchmark refuses to boot when another
serve process or any GPU compute app is live, and it always stops the engine
it started — including on failure, where it still writes what it measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .enginecheck import CheckContext, cross_check, read_boot_log
from .enginerun import (EngineEnv, EngineRunError, ServeProcess,
                        arena_holder_pids, engine_pids, gpu_compute_apps)

# The dossier-4b build precondition the RUNNER must verify: lightning top-k
# admitted contested threshold ties in atomicAdd ARRIVAL ORDER before this
# commit, so KV-set membership depended on warp scheduling.
DET_TOPK_TIES_COMMIT = "8228e71"
KERNELS_SUBMODULE = os.path.join("deps", "LayerStoRmKernels")

# Generation leg: one FIXED prompt across all runs (identity needs identical
# inputs) that is short enough for decode to dominate the wall.
#
# IT MUST BE A CONTINUATION SEED, NOT AN INSTRUCTION (2026-09-01, GF3 ladder
# re-run).  `ServeProcess.complete` posts to /v1/completions and applies NO
# chat template, so the model sees RAW text and continues it as a base LM.
# Given a finished imperative sentence ("Explain, step by step, ... can
# finish."), greedy decoding continues it the only way the surface form
# invites — by restating the instruction — and every sample of a HEALTHY
# engine was discarded at repetition_ratio 0.138 / echo 0.850 while the
# ladder rungs (real prose continuations, same boot) scored 1.0 and read as
# fluent on-topic prose.  A degeneracy gate that a good engine cannot pass
# measures the prompt, not the engine.  So: an UNFINISHED expository
# sentence, with no imperative to echo and nothing quotable in it.
# The proper fix is for the benchmark to speak the model's chat template
# when it wants instruction behaviour — TD-AUTOCONFIG-BENCHMARK-RAW-COMPLETION.
GEN_PROMPT = ("Attention replaced recurrence in sequence models because it "
              "lets every position read every other position directly. "
              "In the years that followed, the main cost of that choice "
              "turned out to be")

# Prompt corpus for the ladder: REAL prose from this checkout, in a fixed
# order, so the ladder prefills a routing distribution that looks like text
# rather than uniform noise over the vocabulary.  Recorded by content hash.
CORPUS_FILES = ("spec/IMPLEMENTATION_GUIDE.md", "spec/SPEC_UPDATES.md",
                "spec/INVARIANTS.md", "spec/PYTHON_ORCHESTRATOR.md",
                "spec/TESTING.md", "spec/CHANGELOG.md", "spec/UPDATES.md",
                "spec/TECH_DEBT.md", "docs/AUTOCONFIG_MODEL.md",
                "spec/AUTO_RUN.md", "README.md")

ORCH_STATS_RE = re.compile(
    r"\[orch-stats\] req=(?P<req>\d+) tokens=(?P<tokens>\d+) "
    r"finish=(?P<finish>\S+) rounds=(?P<rounds>\d+) "
    r"acc=(?P<acc>[\d.]+) \((?P<accepted>\d+)/(?P<proposed>\d+)\) "
    r"prefix_hit=(?P<prefix_hit>\d+) prefill_ms=(?P<prefill_ms>[\d.]+) "
    r"decode_ms=(?P<decode_ms>[\d.]+) decode_tok_s=(?P<decode_tok_s>[\d.]+) "
    r"moe_degraded_layers=(?P<degraded>\d+) "
    r"degraded_retries=(?P<retries>\d+) "
    r"indexer_dense_steps=(?P<dense>\d+)")


class BenchmarkError(RuntimeError):
    """The benchmark could not produce an honest number."""


# --------------------------------------------------------------- options

@dataclass(frozen=True)
class BenchmarkOptions:
    """Everything the benchmark mode can be told.  All of it arrives as CLI
    flags — no config/schema field is read or written for the benchmark
    itself (schema work is serialised in this campaign)."""
    label: str = ""                  # arm name for the chart legend
    ladder: tuple = ()               # explicit prompt lengths (tokens)
    ladder_steps: int = 4            # auto-ladder: how many rungs
    ladder_ratio: int = 2            # auto-ladder: rung spacing
    ladder_floor: int = 1024         # auto-ladder: smallest rung
    repeats: int = 3                 # repetitions per ladder rung
    gen_runs: int = 5                # generation repetitions (GF3.15: 5)
    gen_tokens: int = 300            # tokens per generation run (GF3.15: 300)
    ladder_gen_tokens: int = 8       # tokens each LADDER rung decodes after
                                     # its prefill.  The default 8 is a cheap
                                     # "did it decode at all" tail; raise it
                                     # to measure DECODE AT CONTEXT — the
                                     # generation leg's own prompt is short,
                                     # so it reports decode at ~0 context and
                                     # says nothing about how decode scales
                                     # with sequence length.
    corpus: str = ""                 # token-id or text corpus override
    out_path: str = ""               # results JSON ("" = beside the config)
    request_timeout_s: float = 0.0   # 0 = derive from the prompt length
    boot_timeout_s: float = 2400.0
    skip_ladder: bool = False
    skip_generation: bool = False
    allow_busy_box: bool = False     # override the two-engines guard
    tokenizer_path: str = ""         # override the recipe's tokenizer


# ------------------------------------------------------------ the ladder

def ladder_targets(max_sequence_length: int, gen_tokens: int = 8,
                   steps: int = 4, ratio: int = 2,
                   floor: int = 1024) -> list:
    """Prompt lengths for the prefill ladder, SCALED TO THIS RECIPE.

    Anchored at the top of the recipe's own per-request context and halving
    down: the top rung is what a long-context config exists for, and the
    rungs below it give the scaling curve on the same axis.  Rounded down to
    the 64-token prefill grid so a rung is reproducible across arms.

    The alternative — a fixed 8k/20k/25k ladder — measures the same three
    absolute points on a 25k config and on a 1M config, which says nothing
    about the second one.  Pass ``--benchmark-ladder`` for absolute points
    when continuity with an existing ledger row matters.
    """
    if max_sequence_length <= 0:
        raise BenchmarkError(
            "serving.max_sequence_length is missing or <= 0 in the recipe — "
            "the ladder scales to it and cannot be derived without it "
            "(pass --benchmark-ladder with explicit prompt lengths)")
    top = int((max_sequence_length - max(gen_tokens, 0)) * 0.95)
    out: list = []
    v = top
    for _ in range(max(steps, 1)):
        rung = (v // 64) * 64
        if rung < floor:
            break
        out.append(rung)
        v //= max(ratio, 2)
    if not out:
        raise BenchmarkError(
            f"no ladder rung fits: max_sequence_length={max_sequence_length} "
            f"leaves a top rung below the floor {floor} "
            f"(lower --benchmark-ladder-floor or pass --benchmark-ladder)")
    return sorted(set(out))


# ------------------------------------------------------------- the corpus

def _read_corpus_files(repo_root: str) -> tuple:
    """Concatenated repo prose, in a FIXED order, with its provenance."""
    parts, used = [], []
    for rel in CORPUS_FILES:
        p = os.path.join(repo_root, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip():
            parts.append(text)
            used.append(rel)
    if not parts:
        raise BenchmarkError(
            "no prompt corpus: none of the checked-in prose files "
            f"({', '.join(CORPUS_FILES)}) were readable under {repo_root} "
            "— pass --benchmark-corpus <file>")
    return "\n\n".join(parts), used


@dataclass
class PromptCorpus:
    """A ring of prompt material.  Slices at DIFFERENT offsets share no
    prefix, so repeated ladder samples never hit the prefix cache (which
    would report a ~0 ms prefill and silently invent a number)."""
    ids: list = field(default_factory=list)     # token ids (preferred)
    text: str = ""                              # fallback material
    source: str = ""
    files: tuple = ()
    sha256: str = ""
    chars_per_token: float = 4.0

    @property
    def token_mode(self) -> bool:
        return bool(self.ids)

    def length(self) -> int:
        return len(self.ids) if self.ids else len(self.text)

    def slice(self, offset: int, n_tokens: int):
        """A prompt of about ``n_tokens`` tokens starting at ``offset``.

        Token mode is EXACT.  Text mode is approximate — the recorded
        x-value is always the server's own ``usage.prompt_tokens``, never
        this estimate."""
        total = self.length()
        if total == 0:
            raise BenchmarkError("prompt corpus is empty")
        if self.ids:
            off = offset % total
            return [self.ids[(off + i) % total] for i in range(n_tokens)]
        want = max(int(n_tokens * self.chars_per_token), 1)
        off = offset % total
        if want <= total:
            if off + want <= total:
                return self.text[off:off + want]
            return self.text[off:] + self.text[:off + want - total]
        reps = (want // total) + 2
        return (self.text * reps)[off:off + want]

    def wraps_for(self, n_tokens: int) -> int:
        """How many times a prompt of this size cycles the corpus (0 = the
        corpus covers it outright).  Recorded: a wrapped prompt is
        self-similar material, which the reader must know."""
        total = self.length()
        need = n_tokens if self.ids else int(n_tokens * self.chars_per_token)
        return 0 if total >= need else need // max(total, 1)


def build_corpus(repo_root: str, cfg: dict, opts: BenchmarkOptions,
                 log=print) -> PromptCorpus:
    """The prompt material for the ladder.

    Preference order, loudest first:
      1. ``--benchmark-corpus`` — a whitespace-separated TOKEN-ID file (the
         GATE_CORPUS format the campaign's gates already use) when it parses
         as one, else its raw text;
      2. this checkout's prose, tokenized with the RECIPE's own tokenizer —
         exact prompt lengths, model-correct ids;
      3. this checkout's prose as TEXT (no tokenizer available) — the server
         tokenizes it and reports the real length back.
    """
    if opts.corpus:
        with open(opts.corpus, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]
        toks = raw.split()
        if toks and all(t.lstrip("-").isdigit() for t in toks):
            log(f"autoconfig: benchmark corpus — {len(toks)} token ids from "
                f"{opts.corpus}")
            return PromptCorpus(ids=[int(t) for t in toks],
                                source=f"token-id file {opts.corpus}",
                                files=(opts.corpus,), sha256=digest)
        log(f"autoconfig: benchmark corpus — text from {opts.corpus} "
            f"({len(raw)} chars)")
        return PromptCorpus(text=raw, source=f"text file {opts.corpus}",
                            files=(opts.corpus,), sha256=digest)

    text, files = _read_corpus_files(repo_root)
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    tok_path = opts.tokenizer_path or \
        (cfg.get("serving") or {}).get("tokenizer_path", "")
    weights = (cfg.get("model") or {}).get("weights_path", "")
    model_dir = ""
    if weights:
        w = weights if os.path.isabs(weights) else os.path.join(repo_root,
                                                                weights)
        model_dir = w if os.path.isdir(w) else os.path.dirname(w)
    ids = _try_tokenize(text, tok_path, model_dir, log)
    if ids:
        log(f"autoconfig: benchmark corpus — {len(files)} checked-in prose "
            f"file(s), {len(ids)} tokens (recipe tokenizer; sha {digest})")
        return PromptCorpus(ids=ids, source="repo prose, recipe tokenizer",
                            files=tuple(files), sha256=digest)
    log(f"autoconfig: benchmark corpus — {len(files)} checked-in prose "
        f"file(s), {len(text)} chars sent as TEXT (no usable tokenizer; "
        f"prompt lengths are approximate and the SERVER's reported "
        f"prompt_tokens is what gets recorded; sha {digest})")
    return PromptCorpus(text=text, source="repo prose, text (no tokenizer)",
                        files=tuple(files), sha256=digest)


def _try_tokenize(text: str, tokenizer_path: str, model_dir: str, log):
    """Encode with the recipe's tokenizer, or return [] and SAY WHY.

    CPU-only and in-process: no CUDA is touched (INV-GPU-1)."""
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir))
        from tokenizer import TokenizerWrapper       # noqa: PLC0415
        tw = TokenizerWrapper(tokenizer_path or "auto",
                              model_path=model_dir or None)
        return list(tw.encode(text))
    except Exception as e:                            # noqa: BLE001
        log(f"autoconfig: benchmark — the recipe's tokenizer "
            f"({tokenizer_path or 'auto'}) could not be loaded "
            f"({type(e).__name__}: {e}); falling back to TEXT prompts")
        return []


# ----------------------------------------------------- box / build facts

def shmem_gib() -> float:
    """Host ``Shmem`` in GiB — the arena-holder witness (dossier 2: the
    holder's RSS is not the signal, the shared/Shmem column is)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Shmem:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        pass
    return -1.0


def _git(root: str, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", root, *args],  # noqa: S603,S607
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def det_topk_ties_state(repo_root: str) -> dict:
    """Dossier 4b precondition 3, verified by the RUNNER (it is not
    observable from the orchestrator): is the DET-TOPK-TIES kernel in the
    kernels submodule this checkout builds against?"""
    sub = os.path.join(repo_root, KERNELS_SUBMODULE)
    if not os.path.isdir(sub):
        return {"state": "unknown", "detail":
                f"{KERNELS_SUBMODULE} is not a directory in {repo_root}"}
    # An UNINITIALISED submodule is an empty directory inside the parent
    # work tree, and `git -C` there answers for the PARENT repo — which
    # would report the parent's HEAD as a kernels commit and silently
    # decide a build precondition from the wrong history.  Verified live:
    # a fresh worktree reported the parent commit and called the kernel
    # "absent".  So: only believe git when its top level IS this directory.
    top = _git(sub, "rev-parse", "--show-toplevel")
    if not top or os.path.realpath(top) != os.path.realpath(sub):
        return {"state": "unknown", "detail":
                f"{KERNELS_SUBMODULE} is not a checked-out submodule under "
                f"{repo_root} (git resolved to {top or 'nothing'}) — run "
                f"`git submodule update --init {KERNELS_SUBMODULE}`, or read "
                "the precondition off the checkout the engine was BUILT from"}
    head = _git(sub, "rev-parse", "--short", "HEAD")
    if not head:
        return {"state": "unknown", "detail":
                f"git could not read HEAD of {KERNELS_SUBMODULE}"}
    try:
        rc = subprocess.run(  # noqa: S603,S607
            ["git", "-C", sub, "merge-base", "--is-ancestor",
             DET_TOPK_TIES_COMMIT, "HEAD"],
            capture_output=True, text=True, timeout=30).returncode
    except (OSError, subprocess.SubprocessError) as e:
        return {"state": "unknown", "head": head, "detail": str(e)}
    return {"state": "present" if rc == 0 else "absent", "head": head,
            "required": DET_TOPK_TIES_COMMIT,
            "detail": f"{KERNELS_SUBMODULE} HEAD {head} "
                      f"{'contains' if rc == 0 else 'DOES NOT contain'} "
                      f"{DET_TOPK_TIES_COMMIT}"}


def gpu_inventory() -> list:
    """Which GPUs this number was measured on (chart axis honesty)."""
    try:
        out = subprocess.run(  # noqa: S603,S607
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3:
                gpus.append({"index": int(f[0]), "name": f[1],
                             "memory_total_mib": int(float(f[2]))})
        return gpus
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


# Schema defaults for the knobs a recipe may legally OMIT (config/schema.json).
# Recording a 0 for an absent key would mislabel a chart axis and, for
# max_sequence_length, would refuse a ladder the engine would happily serve.
SCHEMA_DEFAULTS = {
    "serving.max_sequence_length": 32768,
    "serving.max_concurrent_requests": 32,
    "orchestrator.max_batch_size": 64,
    "orchestrator.prefill_chunk_tokens": 64,
    "parallelism.tensor_parallelism": 2,
    "compute.attention_backend": "snapmla",
    "hardware.dcp_indexer_mode": "replicated",
    "hardware.dcp_kv_mode": "sharded",
    "quantization.kv_cache": "fp8_e4m3",
    "quantization.weights": "nvfp4",
}


def _knob(cfg: dict, path: str):
    """A config value with the SCHEMA's default when the key is absent."""
    node = cfg
    for part in path.split("."):
        node = (node or {}).get(part) if isinstance(node, dict) else None
    return SCHEMA_DEFAULTS[path] if node is None else node


def config_knobs(cfg: dict) -> dict:
    """The identifying knobs that travel WITH every number, so a later
    chart can label its axes instead of guessing which arm was which."""
    hw = cfg.get("hardware") or {}
    gpus = hw.get("gpus") or []
    serving = cfg.get("serving") or {}
    spec = cfg.get("speculation") or {}
    draft = ((spec.get("dspark") or {}).get("checkpoint_path") or "")
    default_roles = ["attention", "resident", "expert_streaming"]
    experts = [g for g in gpus
               if "expert_streaming" in (g.get("roles") or default_roles)]
    tp_array = hw.get("tp_array")
    memory = cfg.get("memory") or {}
    return {
        "max_sequence_length": int(_knob(cfg, "serving.max_sequence_length")),
        "max_concurrent_requests": int(
            _knob(cfg, "serving.max_concurrent_requests")),
        "max_batch_size": int(_knob(cfg, "orchestrator.max_batch_size")),
        "prefill_chunk_tokens": int(
            _knob(cfg, "orchestrator.prefill_chunk_tokens")),
        "gpu_count": len(gpus),
        "gpu_types": [g.get("type", "") for g in gpus],
        "tensor_parallelism": int(
            _knob(cfg, "parallelism.tensor_parallelism")),
        "tp_array": tp_array if tp_array is not None else [],
        "expert_parallelism": len(experts),
        "dcp_enabled": bool(hw.get("dcp_enabled", True)),
        "dcp_indexer_mode": _knob(cfg, "hardware.dcp_indexer_mode"),
        "dcp_kv_mode": _knob(cfg, "hardware.dcp_kv_mode"),
        "kv_cache_quant": _knob(cfg, "quantization.kv_cache"),
        "weights_quant": _knob(cfg, "quantization.weights"),
        "kv_tiering": bool((memory.get("kv_tiering") or {}).get(
            "enabled", False)),
        "tiered_prefill": bool((memory.get("kv_tiering") or {}).get(
            "tiered_prefill", False)),
        "nvme_tier": bool((memory.get("nvme_tier") or {}).get(
            "enabled", False)),
        "attention_backend": _knob(cfg, "compute.attention_backend"),
        "speculation_method": spec.get("method", "none"),
        "draft": os.path.basename(draft) if draft else "",
        "has_draft": bool(draft) and spec.get("method", "none") != "none",
        "prefix_cache": bool((serving.get("prefix_cache") or {}).get(
            "enabled", True)),
        "placement_table": bool((memory.get("arena_placement") or {}).get(
            "freq_table", "")),
        "arena_persist": bool((memory.get("arena_attach") or {}).get(
            "persist", False)),
        "calibration_path": os.path.basename(
            (cfg.get("gpu_loader") or {}).get("calibration_path", "")),
        "deterministic_ep_combine_config": bool(
            (cfg.get("compute") or {}).get("deterministic_ep_combine", False)),
    }


# ------------------------------------------------------- the serve log

class OrchStatsTail:
    """Follower over the engine's own stdout log.

    ``[orch-stats]`` is printed BEFORE the result is delivered to the HTTP
    layer (orchestrator._finish is called after the print), so the line for
    a completed request is already on disk when the response arrives; the
    short poll is belt-and-braces against buffering."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.lines: list = []
        self.last_req = -1

    def _pump(self) -> None:
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset = f.tell()
        except OSError:
            return
        self.lines.extend(chunk.splitlines())

    def boot_lines(self) -> list:
        self._pump()
        return list(self.lines)

    def next_stats(self, timeout_s: float = 30.0) -> dict:
        """The next unseen ``[orch-stats]`` record, as plain numbers."""
        deadline = time.time() + timeout_s
        while True:
            self._pump()
            for line in self.lines:
                m = ORCH_STATS_RE.search(line)
                if not m:
                    continue
                req = int(m.group("req"))
                if req <= self.last_req:
                    continue
                self.last_req = req
                d = m.groupdict()
                return {
                    "request_id": req,
                    "tokens": int(d["tokens"]),
                    "finish": d["finish"],
                    "rounds": int(d["rounds"]),
                    "acceptance": float(d["acc"]),
                    "accepted": int(d["accepted"]),
                    "proposed": int(d["proposed"]),
                    "prefix_hit_tokens": int(d["prefix_hit"]),
                    "prefill_ms": float(d["prefill_ms"]),
                    "decode_ms": float(d["decode_ms"]),
                    "decode_tok_s": float(d["decode_tok_s"]),
                    "moe_degraded_layers": int(d["degraded"]),
                    "degraded_retries": int(d["retries"]),
                    "indexer_dense_steps": int(d["dense"]),
                }
            if time.time() >= deadline:
                raise BenchmarkError(
                    "the engine served a request but printed no "
                    f"[orch-stats] line within {timeout_s:.0f}s "
                    f"({self.path}) — without the server's own prefill_ms / "
                    "decode_tok_s there is no honest number to report")
            time.sleep(0.25)


def boot_evidence(lines: list) -> dict:
    """Warm-vs-cold, read off the engine's own boot log.

    The AUTHORITY is the live-prepack counter pair, not the presence of a
    word: ``live prepack: filled N slot(s) ... A adopted warm`` (and the
    ``nothing to build (A adopted warm, ...)`` short circuit) say exactly how
    many slots this boot WROTE and how many it inherited.  A boot that
    adopts 11k warm slots and builds 1k is neither "warm" nor "cold", and
    calling it either mislabels a chart line.

    ``process-private`` is its OWN state, not a flavour of cold: the holder
    was unreachable or its store was left in place, so the engine ran on a
    small private arena.  The 2026-08-31 mapped-KDA verification boot served
    from a 28 GB private arena and its 2.3 tok/s was read as a decode datum
    for a while — that confusion is exactly what this state exists to
    prevent (TD-AUTOCONFIG-SERVE-CMD-DEVICE-ORDER's LIVE PROOF note).
    """
    ev: dict = {"arena": "unknown", "warm_slots_adopted": 0,
                "slots_built": 0, "slots_skipped_full": 0,
                "prepack_gb": 0.0, "prepack_seconds": 0.0,
                "arena_coverage": None,
                "process_private": False, "placement_engaged": False,
                "notices": [], "deterministic_ep_combine_notice": "",
                "subgrid_mid_edge_notice": ""}
    for line in lines:
        m = re.search(r"arena_attach: adopted (\d+) warm slot", line)
        if m:
            ev["warm_slots_adopted"] = max(ev["warm_slots_adopted"],
                                           int(m.group(1)))
            ev["notices"].append(line.strip())
        # The definitive end-of-prepack accounting.
        m = re.search(r"live prepack: filled (\d+) slot\(s\) \(([\d.]+) GB\) "
                      r"in ([\d.]+) s .*?(\d+) adopted warm, "
                      r"(\d+) skipped-full", line)
        if m:
            ev["slots_built"] = int(m.group(1))
            ev["prepack_gb"] = float(m.group(2))
            ev["prepack_seconds"] = float(m.group(3))
            ev["warm_slots_adopted"] = max(ev["warm_slots_adopted"],
                                           int(m.group(4)))
            ev["slots_skipped_full"] = int(m.group(5))
            ev["notices"].append(line.strip())
        m = re.search(r"live prepack: nothing to build \((\d+) adopted warm, "
                      r"(\d+) skipped", line)
        if m:
            ev["slots_built"] = 0
            ev["warm_slots_adopted"] = max(ev["warm_slots_adopted"],
                                           int(m.group(1)))
            ev["slots_skipped_full"] = int(m.group(2))
            ev["notices"].append(line.strip())
        if "live prepack: building" in line or "prepack: packing" in line:
            if "placement ENGAGED" in line:
                ev["placement_engaged"] = True
            ev["notices"].append(line.strip())
        if ("arena_attach: holder unreachable" in line
                or "process-private" in line):
            ev["process_private"] = True
            ev["notices"].append(line.strip())
        if "deterministic_ep_combine is OFF" in line:
            ev["deterministic_ep_combine_notice"] = line.strip()
        if "subgrid_mid_edge is ON" in line:
            ev["subgrid_mid_edge_notice"] = line.strip()

    built, adopted = ev["slots_built"], ev["warm_slots_adopted"]
    # Did the host arena actually COVER the expert set?  `skipped-full` is
    # the count the prepack could not place because every arena was full, so
    # coverage < 1 means this boot served from a SHORT arena and its fetch
    # behaviour (and therefore its tok/s) is arena-starved, not the recipe's.
    have = built + adopted
    total = have + ev["slots_skipped_full"]
    ev["arena_coverage"] = round(have / total, 4) if total else None
    if ev["process_private"]:
        ev["arena"] = "process-private"
    elif adopted and not built:
        ev["arena"] = "warm"
    elif adopted and built:
        ev["arena"] = "partial"
    elif built:
        ev["arena"] = "cold"
    ev["notices"] = ev["notices"][:16]
    return ev


# ------------------------------------------------------------- the guard

def refuse_if_box_busy(allow: bool = False) -> dict:
    """Dossier 1: NEVER run two engines — the second one OOM-kills the box.

    Scans /proc directly (dossier 12: a ``pgrep -f``/``pkill -f`` pattern
    inside a compound command matches the command's OWN wrapper shell) and
    asks the driver which processes hold GPU memory."""
    engines = engine_pids()
    apps = gpu_compute_apps()
    state = {"serve_pids": engines, "gpu_compute_apps": apps,
             "arena_holder_pids": arena_holder_pids(),
             "shmem_gib": round(shmem_gib(), 1)}
    if (engines or apps) and not allow:
        who = []
        if engines:
            who.append("serve process(es) " + ", ".join(map(str, engines)))
        if apps:
            who.append("GPU compute app(s) " + ", ".join(
                f"pid {a['pid']} on gpu {a.get('gpu', '?')} "
                f"({a.get('used_mib', 0)} MiB)" for a in apps))
        raise BenchmarkError(
            "the box is BUSY — " + "; and ".join(who) + ". Never two engines "
            "(dossier 1: the second one OOM-kills the machine). Wait for the "
            "other run, or pass --benchmark-allow-busy-box if you know those "
            "processes do not hold the GPUs this recipe needs")
    return state


# -------------------------------------------------------------- the run

def request_timeout_for(opts: BenchmarkOptions, prompt_tokens: int,
                        max_tokens: int) -> float:
    """A HANG guard, not a performance expectation: floors of 2 tok/s
    prefill and 0.5 tok/s decode, so a healthy run never trips it and a
    wedged one does not block the box forever."""
    if opts.request_timeout_s > 0:
        return opts.request_timeout_s
    return 600.0 + prompt_tokens / 2.0 + max_tokens * 2.0


def _spread_pct(values: list) -> float:
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return 0.0 if med == 0 else 100.0 * (max(values) - min(values)) / med


def summarize_samples(samples: list, key: str) -> dict:
    kept = [s for s in samples if not s["discarded"]]
    vals = [s[key] for s in kept]
    return {
        "values": vals,
        "n": len(vals),
        "median": round(statistics.median(vals), 4) if vals else None,
        "min": round(min(vals), 4) if vals else None,
        "max": round(max(vals), 4) if vals else None,
        "spread_pct": round(_spread_pct(vals), 3) if vals else None,
        "discarded": [{"index": s["index"], "reason": s["discard_reason"]}
                      for s in samples if s["discarded"]],
    }


def _ngrams(words: list, n: int = 4) -> list:
    return [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


def text_health(prompt, completion: str) -> dict:
    """Cheap, tokenizer-free degeneracy metrics on the generated text.

    TD-GLM53-EP4-DEGENERATE-GENERATION: the engine's health counters say
    "nothing was late and nothing fell back to dense"; they cannot say "this
    is a sentence".  Two word-level 4-gram ratios say it well enough to gate
    a benchmark:

      ``repetition_ratio``   distinct 4-grams / 4-grams of the completion.
                             Normal prose sits above ~0.9; a loop
                             ("was a little girl who was a little girl who")
                             collapses toward 4/N.
      ``prompt_echo_ratio``  fraction of the completion's 4-grams that occur
                             VERBATIM in the prompt.  A model copying its own
                             prompt approaches 1.0.  Only computed when the
                             prompt is text (token-id corpora report None).

    Both are reported on every sample so a ledger reader can see them even
    when nothing trips.  ``words`` is the completion length these are
    computed over — the ratios are meaningless on a handful of words.
    """
    words = completion.split()
    grams = _ngrams(words)
    rep = round(len(set(grams)) / len(grams), 4) if grams else None
    echo = None
    if isinstance(prompt, str) and grams:
        pgrams = set(_ngrams(prompt.split()))
        echo = round(sum(1 for g in grams if g in pgrams) / len(grams), 4)
    return {"words": len(words), "repetition_ratio": rep,
            "prompt_echo_ratio": echo}


# Thresholds are deliberately EXTREME so a legitimate continuation of a
# repetitive corpus (code, tables, the ladder's own prose) can never trip
# them: normal text scores > 0.9 repetition_ratio and well under 0.5 echo.
DEGENERATE_MIN_WORDS = 24
DEGENERATE_REPETITION_RATIO = 0.30
DEGENERATE_ECHO_RATIO = 0.75


def discard_reason(stats: dict, health: dict | None = None) -> str:
    """Dossier 4b: a sample computed on an incomplete expert set, one that
    fell back to a dense indexer, or one whose OUTPUT IS NOT TEXT, is not a
    number.  Say so; never average it in."""
    if stats["moe_degraded_layers"]:
        return (f"moe_degraded_layers={stats['moe_degraded_layers']} — the "
                "answer was computed with an INCOMPLETE expert set "
                "(dossier 4b); not comparable and not averaged")
    if stats["indexer_dense_steps"]:
        return (f"indexer_dense_steps={stats['indexer_dense_steps']} — "
                "attention ran with DEAD indexer coverage, a ~10x-slower "
                "BUG witness (TD-INDEXER-NO-DENSE-FALLBACK), not load")
    if health and (health.get("words") or 0) >= DEGENERATE_MIN_WORDS:
        rep = health.get("repetition_ratio")
        echo = health.get("prompt_echo_ratio")
        if rep is not None and rep <= DEGENERATE_REPETITION_RATIO:
            return (f"repetition_ratio={rep} — the completion LOOPS "
                    "(distinct 4-grams / 4-grams); degenerate output is not a "
                    "serving rate, and it inflates the decode rate by routing "
                    "to a repetitive expert set "
                    "(TD-GLM53-EP4-DEGENERATE-GENERATION)")
        if echo is not None and echo >= DEGENERATE_ECHO_RATIO:
            return (f"prompt_echo_ratio={echo} — the completion ECHOES the "
                    "prompt instead of continuing it; degenerate output is "
                    "not a serving rate "
                    "(TD-GLM53-EP4-DEGENERATE-GENERATION)")
    return ""


def weights_abs_for(repo_root: str, cfg: dict) -> str:
    """The recipe's weights path resolved against the data root (checkpoint-
    derived cross-check rows need to read the GGUF headers)."""
    w = (cfg.get("model") or {}).get("weights_path", "")
    for root in (repo_root, "."):
        cand = os.path.join(root, w)
        if w and os.path.exists(cand):
            return cand
    return ""


def run_benchmark(config_path: str, env: EngineEnv, log_dir: str,
                  opts: BenchmarkOptions, log=print) -> dict:
    """Boot the recipe once, measure both legs, stop the engine, return the
    results record.  The engine is ALWAYS stopped — including on failure,
    where whatever was measured is still returned and written."""
    repo_root = os.path.abspath(env.repo_root)
    with open(config_path) as f:
        cfg = json.load(f)

    knobs = config_knobs(cfg)
    box_before = refuse_if_box_busy(opts.allow_busy_box)
    corpus = build_corpus(repo_root, cfg, opts, log)

    targets = list(opts.ladder) if opts.ladder else ladder_targets(
        knobs["max_sequence_length"], gen_tokens=opts.ladder_gen_tokens,
        steps=opts.ladder_steps, ratio=opts.ladder_ratio,
        floor=opts.ladder_floor)
    over = [t for t in targets if t >= knobs["max_sequence_length"]]
    if over:
        raise BenchmarkError(
            f"ladder rung(s) {over} are at or above the recipe's "
            f"serving.max_sequence_length={knobs['max_sequence_length']} — "
            "the server rejects such a prompt with HTTP 400.  Lower "
            "--benchmark-ladder")
    if opts.skip_ladder:
        targets = []

    n_req = (0 if opts.skip_generation else opts.gen_runs) + \
        len(targets) * opts.repeats + 1
    log(f"autoconfig: benchmark — {n_req} request(s): 1 warm-up"
        + ("" if opts.skip_generation else
           f", {opts.gen_runs} x {opts.gen_tokens}-token generation")
        + ("" if not targets else
           f", ladder {targets} x {opts.repeats} repeat(s)"))

    # dossier 4b precondition 1: the runner's own env sets it (EngineEnv
    # .base_env setdefault), and the ENGINE's boot notice is the check.
    det_env = env.base_env().get("LAYERSTORM_DETERMINISTIC_EP_COMBINE", "")
    kernels = det_topk_ties_state(repo_root)
    if kernels["state"] != "present":
        log("autoconfig: benchmark WARNING — dossier 4b precondition 3 is "
            f"NOT satisfied: {kernels['detail']}.  Token-identity across the "
            "generation runs is NOT a valid signal on this build; the tok/s "
            "numbers stand, the identity verdict is recorded as unverifiable")

    os.makedirs(log_dir, exist_ok=True)
    serve_log = os.path.join(log_dir, "benchmark_serve.log")
    proc = ServeProcess(config_path, env, serve_log,
                        tokenizer_path=(opts.tokenizer_path or ""))
    tail = OrchStatsTail(serve_log)

    record: dict = {
        "schema": "layerstorm.autoconfig.benchmark/1",
        "arm": {
            "label": opts.label or os.path.basename(config_path).rsplit(
                ".json", 1)[0],
            "config_path": os.path.abspath(config_path),
            "config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
            "fingerprint": (cfg.get("autoconfig") or {}).get(
                "fingerprint", ""),
            "model": {
                "architecture": (cfg.get("model") or {}).get(
                    "architecture", ""),
                "weights_path": (cfg.get("model") or {}).get(
                    "weights_path", ""),
                "num_hidden_layers": (cfg.get("model") or {}).get(
                    "num_hidden_layers", 0),
                "n_routed_experts": (cfg.get("model") or {}).get(
                    "n_routed_experts", 0),
            },
            "knobs": knobs,
        },
        "conditions": {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_root": repo_root,
            "git_commit": _git(repo_root, "rev-parse", "--short", "HEAD"),
            "git_branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git(repo_root, "status", "--porcelain")),
            "gpus": gpu_inventory(),
            "box_before": box_before,
            "preconditions_4b": {
                "deterministic_ep_combine": (
                    "on" if (det_env and det_env[0] != "0")
                    else "off" if det_env
                    else ("on" if knobs["deterministic_ep_combine_config"]
                          else "off")),
                "deterministic_ep_combine_source": (
                    f"env LAYERSTORM_DETERMINISTIC_EP_COMBINE={det_env}"
                    if det_env else
                    "compute.deterministic_ep_combine in the recipe"),
                "det_topk_ties_kernel": kernels,
                "engine_boot_notice": "",
                "subgrid_mid_edge": "off",
                "degraded_samples": 0,
                "dense_indexer_samples": 0,
                # TD-GLM53-EP4-DEGENERATE-GENERATION: samples the counters
                # called healthy and the TEXT did not.
                "degenerate_text_samples": 0,
            },
            "boot": {},
            "corpus": {
                "source": corpus.source,
                "files": list(corpus.files),
                "sha256": corpus.sha256,
                "tokens": len(corpus.ids) if corpus.token_mode else 0,
                "chars": 0 if corpus.token_mode else len(corpus.text),
                "mode": "token-ids" if corpus.token_mode else "text",
            },
        },
        "generation": {},
        "ladder": [],
        "samples": [],
        "aborted": "",
    }

    idx = 0
    p4 = record["conditions"]["preconditions_4b"]
    try:
        t0 = time.time()
        proc.start()
        proc.wait_ready(opts.boot_timeout_s)
        boot_wall = time.time() - t0

        # Hold the solver's model of the engine against the engine's own boot
        # figures (enginecheck.CHECKS).  Unlike the DERIVATION boots, drift
        # here is RECORDED and reported, not fatal: this mode's contract is
        # "measure what actually booted and never lose the artifacts", and a
        # boot that came up produced valid tok/s regardless of whether the
        # solver predicted its carve.  The recipe it measured is still suspect,
        # so the verdict travels with the results.
        xcheck = cross_check(read_boot_log(serve_log),
                             CheckContext(cfg, weights_abs_for(repo_root, cfg)))
        record["conditions"]["engine_cross_check"] = [
            {"check": r.row.name, "status": r.status,
             "solver": None if r.solver is None else r.solver.value,
             "engine": r.engine, "unit": r.row.unit,
             "solver_site": r.row.solver_site, "engine_site": r.row.engine_site,
             "detail": r.describe()}
            for r in xcheck]
        for r in xcheck:
            if r.status != "agree":
                log("autoconfig: benchmark " + ("DRIFT — " if r.fatal else "")
                    + r.describe().replace("\n", "\n           "))

        ev = boot_evidence(tail.boot_lines())
        record["conditions"]["boot"] = {
            "mode": ("holder-attached" if box_before["arena_holder_pids"]
                     else "no-holder"),
            "holder_pids_before": box_before["arena_holder_pids"],
            "shmem_gib_before": box_before["shmem_gib"],
            "shmem_gib_after_boot": round(shmem_gib(), 1),
            "boot_wall_s": round(boot_wall, 1),
            "arena": ev["arena"],
            "warm_slots_adopted": ev["warm_slots_adopted"],
            "slots_built": ev["slots_built"],
            "slots_skipped_full": ev["slots_skipped_full"],
            "arena_coverage": ev["arena_coverage"],
            "prepack_gb": ev["prepack_gb"],
            "prepack_seconds": ev["prepack_seconds"],
            "placement_engaged": ev["placement_engaged"],
            "evidence": ev["notices"][:12],
            "log": serve_log,
        }
        if ev["deterministic_ep_combine_notice"]:
            p4["deterministic_ep_combine"] = "off"
            p4["engine_boot_notice"] = ev["deterministic_ep_combine_notice"]
        if ev["subgrid_mid_edge_notice"]:
            p4["subgrid_mid_edge"] = "on"
            p4["engine_boot_notice"] = ev["subgrid_mid_edge_notice"]
        log(f"autoconfig: benchmark — engine ready in {boot_wall:.1f}s "
            f"(arena {ev['arena']}, {ev['warm_slots_adopted']} warm slot(s) "
            f"adopted); deterministic_ep_combine="
            f"{p4['deterministic_ep_combine']}, DET-TOPK-TIES="
            f"{kernels['state']}")

        model_id = proc.model_id()

        def one(prompt, max_tokens: int, kind: str, target: int,
                repeat: int) -> dict:
            nonlocal idx
            idx += 1
            n_est = len(prompt) if isinstance(prompt, list) else target
            tmo = request_timeout_for(opts, n_est or 64, max_tokens)
            wall0 = time.time()
            resp = proc.complete(model_id, prompt, max_tokens, tmo)
            wall = time.time() - wall0
            st = tail.next_stats()
            usage = resp.get("usage") or {}
            gen_text = (resp.get("choices") or [{}])[0].get("text", "")
            health = text_health(prompt, gen_text)
            reason = discard_reason(st, health)
            s = {
                "index": idx,
                "kind": kind,
                "target_tokens": target,
                "repeat": repeat,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "client_wall_s": round(wall, 3),
                "text": gen_text,
                "discarded": bool(reason),
                "discard_reason": reason,
            }
            s.update(st)
            s.update(health)
            fresh = s["prompt_tokens"] - s["prefix_hit_tokens"]
            s["prefill_tok_s"] = round(
                1000.0 * max(fresh - 1, 0) / st["prefill_ms"], 4) \
                if st["prefill_ms"] > 0 else 0.0
            if st["moe_degraded_layers"]:
                p4["degraded_samples"] += 1
            if st["indexer_dense_steps"]:
                p4["dense_indexer_samples"] += 1
            if reason and not st["moe_degraded_layers"] \
                    and not st["indexer_dense_steps"]:
                p4["degenerate_text_samples"] += 1
            record["samples"].append(s)
            tag = kind + (f"@{target}" if target else "")
            log(f"autoconfig:   [{idx}/{n_req}] {tag} rep{repeat}: "
                f"prompt={s['prompt_tokens']} "
                f"prefill={st['prefill_ms']:.0f}ms "
                f"({s['prefill_tok_s']:.2f} tok/s) "
                f"decode={st['decode_tok_s']:.3f} tok/s"
                + (f"  DISCARDED: {reason}" if reason else ""))
            return s

        # Warm-up: the FIRST request of a boot may finalize MoE layers
        # degraded (TD-V4-FIRSTREQ-COLD-SHARE-OVER-CAPACITY) and is never a
        # control (dossier 4b / tools/serve_determinism_probe.py).
        one(corpus.slice(0, 256), 8, "warmup", 0, 0)

        if not opts.skip_generation:
            gsamples = [one(GEN_PROMPT, opts.gen_tokens, "generation", 0,
                            k + 1) for k in range(opts.gen_runs)]
            kept = [s for s in gsamples if not s["discarded"]]
            texts = {s["text"] for s in kept}
            record["generation"] = {
                "prompt": GEN_PROMPT,
                "prompt_tokens": kept[0]["prompt_tokens"] if kept else 0,
                "max_tokens": opts.gen_tokens,
                "greedy": True,
                "batch_size": 1,
                "runs": opts.gen_runs,
                "tok_s": summarize_samples(gsamples, "decode_tok_s"),
                "identical_outputs": (len(texts) == 1 and len(kept) > 1),
                "identity_valid": (kernels["state"] == "present"
                                   and p4["deterministic_ep_combine"] == "on"
                                   and p4["subgrid_mid_edge"] == "off"
                                   and len(kept) > 1),
                "prefix_hit_tokens": [s["prefix_hit_tokens"]
                                      for s in gsamples],
            }

        for target in targets:
            lsamples = []
            for r in range(opts.repeats):
                # A DIFFERENT ring offset per repeat: two slices that share
                # no prefix cannot hit the prefix cache, which would report
                # a ~0 ms prefill and invent a number.
                off = 1 + (r + 1) * 8191 + target
                lsamples.append(one(corpus.slice(off, target),
                                    opts.ladder_gen_tokens, "prefill",
                                    target, r + 1))
            record["ladder"].append({
                "target_tokens": target,
                "fraction_of_max_sequence_length": round(
                    target / knobs["max_sequence_length"], 4)
                if knobs["max_sequence_length"] else None,
                "prompt_tokens": [s["prompt_tokens"] for s in lsamples],
                "corpus_wraps": corpus.wraps_for(target),
                "repeats": opts.repeats,
                "prefill_tok_s": summarize_samples(lsamples, "prefill_tok_s"),
                "prefill_ms": summarize_samples(lsamples, "prefill_ms"),
                # Decode measured AT this context depth (max_tokens =
                # --benchmark-ladder-gen-tokens).  This is the series that
                # answers "how does generation scale with context?" — the
                # generation leg alone cannot, its prompt is ~50 tokens.
                "decode_gen_tokens": opts.ladder_gen_tokens,
                "decode_tok_s": summarize_samples(lsamples, "decode_tok_s"),
                "prefix_hit_tokens": [s["prefix_hit_tokens"]
                                      for s in lsamples],
            })
    except (BenchmarkError, EngineRunError, OSError, ValueError) as e:
        record["aborted"] = f"{type(e).__name__}: {e}"
        log(f"autoconfig: benchmark ABORTED — {record['aborted']} "
            f"(partial results are still written)")
    finally:
        # The contract: this mode never leaves an engine serving.
        rc = proc.stop()
        boot = record["conditions"].setdefault("boot", {})
        boot["stop_returncode"] = rc
        boot["shmem_gib_after_stop"] = round(shmem_gib(), 1)
        record["conditions"]["finished_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        still = engine_pids()
        record["conditions"]["engines_still_running"] = still
        if still:
            log(f"autoconfig: benchmark WARNING — serve process(es) {still} "
                "are STILL running after the stop; investigate before the "
                "next boot (dossier 1: never two engines)")
    return record


# ------------------------------------------------------------- artifacts

CSV_KNOB_COLUMNS = (
    "max_sequence_length", "max_concurrent_requests", "max_batch_size",
    "gpu_count", "tensor_parallelism", "expert_parallelism",
    "dcp_indexer_mode", "dcp_kv_mode", "kv_tiering", "has_draft",
    "speculation_method", "attention_backend", "prefix_cache",
    "placement_table",
)

CSV_HEADER = (
    ["arm_label", "config_sha256", "git_commit", "measurement",
     "target_tokens", "fraction_of_max", "repeat", "prompt_tokens",
     "prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s",
     "prefix_hit_tokens", "moe_degraded_layers", "indexer_dense_steps",
     "discarded", "arena", "boot_mode", "boot_wall_s",
     "deterministic_ep_combine", "det_topk_ties_kernel"]
    + list(CSV_KNOB_COLUMNS))


def to_csv_rows(record: dict) -> list:
    """LONG format, one row per REQUEST, with the arm's knobs repeated on
    every row — so several arms' CSVs concatenate into one chartable table
    with honest axis labels and no join step."""
    arm = record["arm"]
    cond = record["conditions"]
    boot = cond.get("boot") or {}
    p4 = cond.get("preconditions_4b") or {}
    knobs = arm["knobs"]
    maxseq = knobs.get("max_sequence_length") or 0
    rows = [list(CSV_HEADER)]
    for s in record.get("samples", []):
        if s["kind"] == "warmup":
            continue
        rows.append([
            arm["label"], arm["config_sha256"], cond.get("git_commit", ""),
            "prefill" if s["kind"] == "prefill" else "generation",
            s["target_tokens"],
            round(s["target_tokens"] / maxseq, 4)
            if maxseq and s["target_tokens"] else "",
            s["repeat"], s["prompt_tokens"],
            round(s["prefill_ms"], 3), s["prefill_tok_s"],
            round(s["decode_ms"], 3), s["decode_tok_s"],
            s["prefix_hit_tokens"], s["moe_degraded_layers"],
            s["indexer_dense_steps"], int(s["discarded"]),
            boot.get("arena", ""), boot.get("mode", ""),
            boot.get("boot_wall_s", ""),
            p4.get("deterministic_ep_combine", ""),
            (p4.get("det_topk_ties_kernel") or {}).get("state", ""),
        ] + [knobs.get(k, "") for k in CSV_KNOB_COLUMNS])
    return rows


def write_csv(path: str, record: dict) -> None:
    import csv                                        # noqa: PLC0415
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(to_csv_rows(record))


def ledger_lines(record: dict) -> str:
    """A ready-to-paste ``spec/measurements/`` row (CLAUDE.md: real perf
    numbers belong in the per-domain ledger, with timestamp + regime/arm +
    number + source)."""
    arm = record["arm"]
    cond = record["conditions"]
    boot = cond.get("boot") or {}
    p4 = cond.get("preconditions_4b") or {}
    k = arm["knobs"]
    date = cond.get("started_utc", "")[:10]
    regime = (f"tp={k.get('tensor_parallelism')} "
              f"ep={k.get('expert_parallelism')} "
              f"{k.get('gpu_count')} GPU(s), "
              f"max_seq={k.get('max_sequence_length')}, "
              f"conc={k.get('max_concurrent_requests')}, "
              f"kv={k.get('dcp_kv_mode')}/indexer={k.get('dcp_indexer_mode')}, "
              f"tiering={'ON' if k.get('kv_tiering') else 'OFF'}, "
              f"draft={k.get('draft') or 'none'}")
    pre = (f"det_ep_combine={p4.get('deterministic_ep_combine')}, "
           f"DET-TOPK-TIES="
           f"{(p4.get('det_topk_ties_kernel') or {}).get('state')}, "
           f"degraded={p4.get('degraded_samples')}, "
           f"dense_steps={p4.get('dense_indexer_samples')}, "
           f"degenerate_text={p4.get('degenerate_text_samples')}, "
           f"arena={boot.get('arena')} ({boot.get('mode')}, "
           f"boot {boot.get('boot_wall_s')} s)")
    out = [f"- {date} autoconfig benchmark, arm **{arm['label']}** "
           f"({regime}); {pre}. Source: {arm['config_path']}, "
           f"{boot.get('log', '')}, commit {cond.get('git_commit', '')}"
           + (" [DIRTY TREE]" if cond.get("git_dirty") else "") + "."]
    gen = record.get("generation") or {}
    if (gen.get("tok_s") or {}).get("n"):
        t = gen["tok_s"]
        out.append(
            f"  - decode B=1 greedy {gen['max_tokens']}-token: "
            f"**{t['median']} tok/s median** (n={t['n']}: "
            + "/".join(f"{v:.3f}" for v in t["values"])
            + f"; spread {t['spread_pct']}%); run-to-run outputs "
            + ("IDENTICAL" if gen.get("identical_outputs") else "DIFFER")
            + ("" if gen.get("identity_valid")
               else " (identity NOT gate-valid: 4b preconditions unmet)")
            + ".")
    for rung in record.get("ladder", []):
        t = rung["prefill_tok_s"]
        if not t.get("n"):
            continue
        d = rung.get("decode_tok_s") or {}
        out.append(
            f"  - served prefill @{rung['target_tokens']} tok "
            f"({rung['fraction_of_max_sequence_length']} of max_seq): "
            f"**{t['median']} tok/s median** (n={t['n']}: "
            + "/".join(f"{v:.2f}" for v in t["values"])
            + f"; spread {t['spread_pct']}%)"
            + (f"; decode AT that context "
               f"**{d['median']} tok/s median** over "
               f"{rung.get('decode_gen_tokens')} token(s) (n={d['n']}: "
               + "/".join(f"{v:.3f}" for v in d["values"]) + ")"
               if d.get("n") else "") + ".")
    if record.get("aborted"):
        out.append(f"  - RUN ABORTED: {record['aborted']} — the rows above "
                   "are what completed before the abort.")
    return "\n".join(out) + "\n"


def write_artifacts(record: dict, out_json: str, log=print) -> dict:
    base = out_json[:-5] if out_json.endswith(".json") else out_json
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".",
                exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    csv_path = base + ".csv"
    write_csv(csv_path, record)
    ledger_path = base + ".ledger.md"
    with open(ledger_path, "w") as f:
        f.write(ledger_lines(record))
    log(f"autoconfig: benchmark — wrote {out_json}")
    log(f"autoconfig: benchmark — wrote {csv_path} (chartable long format; "
        "concatenate several arms' CSVs to compare them)")
    log(f"autoconfig: benchmark — wrote {ledger_path} (paste into the "
        "matching spec/measurements/ ledger)")
    return {"json": out_json, "csv": csv_path, "ledger": ledger_path}


def summarize(record: dict, log=print) -> None:
    gen = record.get("generation") or {}
    t = gen.get("tok_s") or {}
    log("")
    log(f"autoconfig: BENCHMARK — arm {record['arm']['label']}")
    if t.get("n"):
        log(f"autoconfig:   generation B=1 greedy: {t['median']} tok/s "
            f"median of {t['n']} ("
            + "/".join(f"{v:.3f}" for v in t["values"])
            + f"; spread {t['spread_pct']}%), outputs "
            + ("IDENTICAL" if gen.get("identical_outputs") else "DIFFER"))
    elif not gen:
        log("autoconfig:   generation: SKIPPED")
    else:
        log("autoconfig:   generation: NO VALID SAMPLE (all discarded)")
    for rung in record.get("ladder", []):
        p = rung["prefill_tok_s"]
        if p.get("n"):
            d = rung.get("decode_tok_s") or {}
            log(f"autoconfig:   prefill @{rung['target_tokens']:>8} tok: "
                f"{p['median']} tok/s median of {p['n']} ("
                + "/".join(f"{v:.2f}" for v in p["values"])
                + f"; spread {p['spread_pct']}%)"
                + (f"   decode@ctx {d['median']} tok/s"
                   f" (n={d['n']}, {rung.get('decode_gen_tokens')} tok)"
                   if d.get("n") else ""))
        else:
            log(f"autoconfig:   prefill @{rung['target_tokens']:>8} tok: "
                "NO VALID SAMPLE (all discarded)")
    p4 = record["conditions"]["preconditions_4b"]
    boot = record["conditions"].get("boot") or {}
    log(f"autoconfig:   conditions: arena={boot.get('arena')} "
        f"({boot.get('mode')}, boot {boot.get('boot_wall_s')} s), "
        f"det_ep_combine={p4['deterministic_ep_combine']}, "
        f"DET-TOPK-TIES="
        f"{(p4.get('det_topk_ties_kernel') or {}).get('state')}, "
        f"degraded={p4['degraded_samples']}, "
        f"dense_steps={p4['dense_indexer_samples']}, "
        f"degenerate_text={p4['degenerate_text_samples']}")
    if p4["degraded_samples"]:
        log("autoconfig:   NOTE: degraded sample(s) were DISCARDED from "
            "every aggregate above (dossier 4b) — see \"discarded\" in the "
            "JSON")
    if record.get("aborted"):
        log(f"autoconfig:   ABORTED: {record['aborted']}")
