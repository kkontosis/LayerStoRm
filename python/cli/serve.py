"""CLI serve — the LayerStoRm serving entry point (#71).

Boots the whole stack in one go:

    C++ engine (layerstorm_engine.start_engine)
      → successor orchestrator (orchestrator.orchestrator.Orchestrator —
        the bridge-based champion decode core: FAR burst sweeps + REEF
        service placement + DSpark overlap speculation)
        → OpenAI-compatible HTTP API (server.http_server.LayerStoRmServer)

with the tokenizer + chat template wired from the model directory, so a
user can ``curl /v1/chat/completions`` against a live model:

    python python/cli/serve.py --config config/my_model.json

Threading model:
  - main thread:      Orchestrator.run() — serves queued requests
                      serially (B=1; TD-ORCH-B-GT-1).  Signal handlers
                      run here, so SIGINT/SIGTERM cleanly flip the
                      shutdown flag.
  - daemon thread:    C++ engine (spawned inside start_engine; never
                      acquires the GIL — the bridge's Cython waits
                      release it while spinning).
  - uvicorn thread:   HTTP server (LayerStoRmServer.run).  Handlers submit
                      InferenceRequests into the orchestrator's
                      thread-safe request deque and wait on callbacks.

The tokenizer is the ONLY component with model-vocabulary knowledge: the
engine works purely in token ids, so eos_token_ids / vocab_size /
think-token ids are detected here and injected into the loop metadata
(stop criteria) and the HTTP server (encode/decode, template, EOS strip).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── sys.path bootstrap ──────────────────────────────────────────────────────
# Works when invoked as `python python/cli/serve.py`, `python -m cli.serve`
# (with python/ already on the path), or imported from tests.  Adds the
# python/ package root and the compiled pybind11 module directory.
_PYTHON_DIR = Path(__file__).resolve().parent.parent          # .../python
_PROJECT_ROOT = _PYTHON_DIR.parent
for _p in (str(_PYTHON_DIR), str(_PROJECT_ROOT / "build" / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orchestrator.dspark_draft import DsparkDraftConfig  # noqa: E402
from orchestrator.orchestrator import Orchestrator  # noqa: E402
from server.http_server import LayerStoRmServer  # noqa: E402
from tokenizer import ChatTemplateRenderer, TokenizerWrapper  # noqa: E402
from tokenizer import locator  # noqa: E402

log = logging.getLogger("layerstorm.serve")

# Files whose presence marks a usable HuggingFace tokenizer directory
# (canonical list + shared search precedence live in tokenizer.locator).
_TOKENIZER_MARKERS = locator.TOKENIZER_MARKERS


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class ServeOptions:
    """Resolved serving options: config `serving` section + CLI overrides."""
    config_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = ""
    max_concurrent: int = 32
    max_queued_requests: int = 16
    sse_heartbeat_seconds: float = 15.0
    max_sequence_length: int = 32768
    tokenizer_path: str = "auto"
    # vLLM-parity serving parsers ("" = disabled): named tool-call parser
    # (server.tool_parsers registry), auto tool choice gating, and named
    # reasoning parser (server.reasoning_parsers registry).
    tool_call_parser: str = ""
    enable_auto_tool_choice: bool = False
    reasoning_parser: str = ""
    # vLLM-parity serving tokenizer mode ("auto" resolves from
    # model.architecture: deepseek_v4 → "deepseek_v4"; else "hf" legacy)
    # and ReasoningConfig JSON passthrough (marker-string overrides for
    # the named reasoning parser; "" = none).
    tokenizer_mode: str = "auto"
    reasoning_config: str = ""
    # None = derive from the config's speculation section (dspark →
    # speculative_tokens); explicit 0 forces plain greedy AR decode.
    speculation_depth: int | None = None
    cycle_budget_us: float = 200.0
    use_test_engine: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="layerstorm-serve",
        description="Serve a model over an OpenAI-compatible HTTP API "
                    "(engine + orchestrator + tokenizer in one process).",
    )
    p.add_argument("--config", default="",
                   help="engine config JSON (config/schema.json). Required "
                        "unless --autoconfig --model derives one (P-31); "
                        "with both, --config carries extras into the "
                        "derivation (base recipe)")
    p.add_argument("--model", default="",
                   help="with --autoconfig: derive the recipe from the "
                        "WEIGHTS themselves (a .gguf file, a dir of GGUF "
                        "shards, or a HF model dir) — no --config needed. "
                        "Measured artifacts beside the weights (trained "
                        "loader calibration, arena placement table) are "
                        "reused; the derive is CPU-only "
                        "(TD-AUTOCONFIG-NO-GLM5NEXT-PROFILE consumer)")
    p.add_argument("--accuracy", default=None,
                   choices=["compact", "standard", "high", "superior"],
                   help="with --autoconfig: the numerics FLOOR (autoconfig "
                        "lever 4, AUTOCONFIG §2.4)")
    p.add_argument("--prefer", default=None,
                   choices=["speed", "balanced", "capacity"],
                   help="with --autoconfig: what the fit gives up FIRST "
                        "(autoconfig lever 3, AUTOCONFIG §2.3; order only)")
    p.add_argument("--host", default=None,
                   help="bind address (overrides serving.host)")
    p.add_argument("--port", type=int, default=None,
                   help="port (overrides serving.port)")
    p.add_argument("--model-name", default=None,
                   help="model id reported by /v1/models "
                        "(default: weights file/dir stem)")
    p.add_argument("--tokenizer-path", default=None,
                   help="HuggingFace tokenizer directory "
                        "(overrides serving.tokenizer_path; 'auto' = "
                        "detect next to model.weights_path)")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="max simultaneous requests "
                        "(overrides serving.max_concurrent_requests)")
    p.add_argument("--max-queued-requests", type=int, default=None,
                   help="max requests WAITING for a generation slot "
                        "(bounded FIFO queue; beyond it → 503 + "
                        "Retry-After; overrides "
                        "serving.max_queued_requests)")
    p.add_argument("--sse-heartbeat-seconds", type=float, default=None,
                   help="SSE comment-keepalive cadence (seconds) while a "
                        "stream has no data (long prefills); 0 disables "
                        "(overrides serving.sse_heartbeat_seconds)")
    p.add_argument("--max-sequence-length", type=int, default=None,
                   help="max prompt tokens "
                        "(overrides serving.max_sequence_length)")
    p.add_argument("--tool-call-parser", default=None,
                   help="named tool-call parser (e.g. glm47) — parses "
                        "model output into OpenAI tool_calls "
                        "(overrides serving.tool_call_parser)")
    p.add_argument("--enable-auto-tool-choice", action="store_const",
                   const=True, default=None,
                   help="enable auto tool choice: run the tool-call parser "
                        "when the request has tools and tool_choice is "
                        "absent/'auto' (requires --tool-call-parser; "
                        "overrides serving.enable_auto_tool_choice)")
    p.add_argument("--reasoning-parser", default=None,
                   help="named reasoning parser (e.g. glm45, deepseek_v4) "
                        "— splits reasoning_content from content "
                        "(overrides serving.reasoning_parser)")
    p.add_argument("--reasoning-config", default=None,
                   help="JSON reasoning config (vLLM ReasoningConfig "
                        "field names: reasoning_start_str/"
                        "reasoning_end_str) — marker overrides for the "
                        "named reasoning parser; empty markers = "
                        "structural boundary "
                        "(overrides serving.reasoning_config)")
    p.add_argument("--tokenizer-mode", default=None,
                   choices=["auto", "hf", "deepseek_v4", "glm5_next"],
                   help="serving tokenizer mode (vLLM parity subset): "
                        "'auto' resolves from model.architecture "
                        "(deepseek_v4 models → deepseek_v4, glm5_next "
                        "models → glm5_next), 'hf' forces the legacy "
                        "behavior, 'deepseek_v4' switches chat-template "
                        "kwarg normalization + thinking defaults to the "
                        "DeepSeek-V4 rules, 'glm5_next' to the GLM-5.3 "
                        "rules (reasoning_effort low/high/max default "
                        "max; clear_thinking default true for chat; "
                        "thinking always on; generation_config.json "
                        "sampling defaults) "
                        "(overrides serving.tokenizer_mode)")
    p.add_argument("--speculation-depth", type=int, default=None,
                   help="speculative decode depth; default derives from "
                        "the config speculation section (0 = plain greedy)")
    p.add_argument("--cycle-budget-us", type=float, default=200.0,
                   help="orchestrator cycle budget in µs (idle throttle)")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.add_argument("--test-engine", action="store_true",
                   help="start the engine with null backends (no CUDA) — "
                        "smoke-testing the serve stack only")
    p.add_argument("--autoconfig", action="store_true",
                   help="OPT-IN hardware-fit derivation (TD-AUTOCONFIG-"
                        "HARDWARE-FIT): detect the hardware, derive a recipe "
                        "from the config's autoconfig levers, persist it next "
                        "to --config, and serve THAT recipe. Reuses the "
                        "persisted recipe while the hardware fingerprint "
                        "matches. Without this flag, autoconfig.enabled=true "
                        "in the config re-derives/serves IN PLACE only when "
                        "its output_path is the --config file itself; a "
                        "differing output_path REFUSES rather than silently "
                        "booting another file (TD-AUTOCONFIG-SERVE-"
                        "SUBSTITUTES-THE-CONFIG)")
    p.add_argument("--autoconfig-redetect", action="store_true",
                   help="with --autoconfig: re-derive even if the stored "
                        "hardware fingerprint still matches")
    return p


def _resolve_tokenizer_mode(mode: str, architecture: str) -> str:
    """vLLM tokenizer-mode auto-default rule (ref/vllm
    vllm/config/model.py:617): "auto" resolves to the model-family mode
    for architectures that have one ("deepseek_v4", "glm5_next"), else
    the legacy behavior ("hf")."""
    if mode == "auto":
        if architecture in ("deepseek_v4", "glm5_next"):
            return architecture
        return "hf"
    return mode


def _parse_reasoning_config(raw: str) -> dict:
    """Parse the --reasoning-config JSON passthrough ("" → {})."""
    if not raw:
        return {}
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise ValueError("reasoning_config must be a JSON object")
    return cfg


def load_config(config_path: str | Path) -> dict:
    """Load the engine config JSON (schema: config/schema.json)."""
    with open(config_path) as f:
        return json.load(f)


def resolve_options(config: dict, args: argparse.Namespace) -> ServeOptions:
    """Merge the config `serving` section with CLI overrides (CLI wins)."""
    serving = config.get("serving") or {}

    def pick(cli_val: Any, cfg_key: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        return serving.get(cfg_key, default)

    model_name = args.model_name
    if not model_name:
        weights = (config.get("model") or {}).get("weights_path") or ""
        p = Path(weights)
        # Strip only known weight-file extensions — model DIRECTORIES may
        # legitimately contain dots (e.g. "DeepSeek-V3.2").
        name = p.name
        if p.suffix.lower() in (".gguf", ".safetensors", ".bin", ".json"):
            name = p.stem
        model_name = name or "layerstorm"

    return ServeOptions(
        config_path=args.config,
        host=pick(args.host, "host", "0.0.0.0"),
        port=int(pick(args.port, "port", 8000)),
        model_name=model_name,
        max_concurrent=int(pick(args.max_concurrent,
                                "max_concurrent_requests", 32)),
        max_queued_requests=int(pick(getattr(args, "max_queued_requests",
                                             None),
                                     "max_queued_requests", 16)),
        sse_heartbeat_seconds=float(pick(getattr(args,
                                                 "sse_heartbeat_seconds",
                                                 None),
                                         "sse_heartbeat_seconds", 15.0)),
        max_sequence_length=int(pick(args.max_sequence_length,
                                     "max_sequence_length", 32768)),
        tokenizer_path=pick(args.tokenizer_path, "tokenizer_path", "auto"),
        tool_call_parser=str(pick(args.tool_call_parser,
                                  "tool_call_parser", "")),
        enable_auto_tool_choice=bool(pick(args.enable_auto_tool_choice,
                                          "enable_auto_tool_choice", False)),
        reasoning_parser=str(pick(args.reasoning_parser,
                                  "reasoning_parser", "")),
        tokenizer_mode=_resolve_tokenizer_mode(
            str(pick(getattr(args, "tokenizer_mode", None),
                     "tokenizer_mode", "auto")),
            (config.get("model") or {}).get("architecture", "") or ""),
        reasoning_config=str(pick(getattr(args, "reasoning_config", None),
                                  "reasoning_config", "")),
        speculation_depth=args.speculation_depth,
        cycle_budget_us=float(args.cycle_budget_us),
        use_test_engine=bool(getattr(args, "test_engine", False)),
    )


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------

def resolve_tokenizer_dir(
    weights_path: str | Path, tokenizer_path: str = "auto",
    model_name: str = "", repo_root: str | Path = "",
) -> Path:
    """Resolve the HuggingFace tokenizer directory for a model.

    P-34 step 1 precedence (tokenizer.locator is the shared search):
    explicit ``tokenizer_path`` (anything but "auto") always wins; "auto"
    looks in the weights directory (or a single-file checkpoint's parent),
    then at name-prefix siblings beside the weights, and — FALLBACK ONLY,
    logged loudly — at repo ``test-data/`` dirs matching ``model_name``.
    """
    if tokenizer_path and tokenizer_path != "auto":
        p = Path(tokenizer_path)
        if not p.exists():
            raise FileNotFoundError(
                f"serving.tokenizer_path does not exist: {p}")
        return p

    root = str(repo_root) if repo_root else str(
        Path(__file__).resolve().parents[2])
    found, tier = locator.locate_tokenizer_dir(
        str(weights_path), model_name=model_name, repo_root=root)
    if found:
        if tier == locator.TIER_TEST_DATA:
            log.warning(
                "TOKENIZER FALLBACK: no tokenizer files beside the weights "
                "(%s) — falling back to repo test-data at %s. This works on "
                "this checkout only; place the HF tokenizer files "
                "(tokenizer.json, tokenizer_config.json, ...) next to the "
                "weights or set serving.tokenizer_path.",
                weights_path, found)
        elif tier == locator.TIER_SIBLING:
            log.info("tokenizer resolved from weights sibling %s", found)
        return Path(found)
    w = Path(weights_path)
    candidate = w if w.is_dir() else w.parent
    raise FileNotFoundError(
        f"no HuggingFace tokenizer files ({', '.join(_TOKENIZER_MARKERS)}) "
        f"found in {candidate}, its name-prefix siblings, or repo "
        f"test-data/ — put the tokenizer files next to the weights or set "
        "serving.tokenizer_path (GGUF-embedded tokenizers are not "
        "extracted; see TD-SERVE-GGUF-TOKENIZER)")


def _has_chat_template(d: Path) -> bool:
    """A directory can carry the chat template as a .jinja file OR
    embedded in tokenizer_config.json (ChatTemplateRenderer reads both)."""
    if (d / "chat_template.jinja").is_file():
        return True
    try:
        with open(d / "tokenizer_config.json") as f:
            return bool(json.load(f).get("chat_template"))
    except (OSError, ValueError):
        return False


def resolve_asset_dir(
    filename: str, tok_dir: Path, weights_path: str | Path,
    model_name: str = "", repo_root: str | Path = "",
    present=None,
) -> Path:
    """P-34: every model-metadata asset (chat_template.jinja,
    generation_config.json, ...) follows the SAME precedence as the
    tokenizer.  The resolved tokenizer dir wins when it already carries
    the asset (an explicit serving.tokenizer_path keeps its assets
    together); otherwise search beside the weights, with repo test-data/
    as a loudly-logged fallback.  Returns the directory to read the asset
    from — tok_dir when the asset exists nowhere (consumers handle
    absence themselves)."""
    check = present if present is not None else (
        lambda d: (Path(d) / filename).is_file())
    if check(tok_dir):
        return Path(tok_dir)
    root = str(repo_root) if repo_root else str(
        Path(__file__).resolve().parents[2])
    d, tier = locator._search_chain(
        str(weights_path), lambda x: check(Path(x)),
        model_name=model_name, repo_root=root)
    if not d:
        return Path(tok_dir)
    if tier == locator.TIER_TEST_DATA:
        log.warning(
            "METADATA FALLBACK: %s not found beside the weights (%s) — "
            "using repo test-data at %s. This works on this checkout "
            "only; place the file next to the weights.",
            filename, weights_path, d)
    return Path(d)


def read_sampling_defaults(
    tokenizer_dir: str | Path, tokenizer_mode: str,
) -> dict:
    """Model-recommended sampling defaults for the serving layer.

    glm5_next only (GF3.13): GLM-5.3-Flash's generation_config.json
    recommends temperature 1.0 / top_p 0.95.  The server applies ONLY
    top_p, and only to requests that explicitly opt into sampling
    (explicit temperature > 0) while leaving top_p unset — the
    greedy-champion routing of unspecified temperature is deliberately
    unchanged, and every other tokenizer mode returns {} so their
    serving paths stay byte-identical.
    """
    if tokenizer_mode != "glm5_next":
        return {}
    path = Path(tokenizer_dir) / "generation_config.json"
    try:
        with open(path) as f:
            gen = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        log.warning("cannot read %s — no sampling defaults applied", path)
        return {}
    defaults = {}
    top_p = gen.get("top_p")
    if isinstance(top_p, (int, float)) and 0.0 < float(top_p) <= 1.0:
        defaults["top_p"] = float(top_p)
    return defaults


# ---------------------------------------------------------------------------
# Speculation derivation (basics: dspark passthrough + explicit depth)
# ---------------------------------------------------------------------------

def derive_speculation(
    config: dict, requested_depth: int | None,
) -> tuple[int, DsparkDraftConfig | None]:
    """Derive (speculation_depth, dspark_config) from the engine config.

    The engine config already selected + loaded the drafter C++-side
    (speculation.method); the Python loop must arm the MATCHING planner.
    Basics scope: dspark (whole-block drafter) and prompt-lookup (any
    depth > 0 with no drafter config).  MTP/self-spec CLI wiring is
    TD-SERVE-SPECULATION.
    """
    spec = config.get("speculation") or {}
    enabled = bool(spec.get("enabled", False))
    method = spec.get("method", "")

    if enabled and method == "dspark":
        ds = spec.get("dspark") or {}
        spec_tokens = int(ds.get("speculative_tokens",
                                 max(1, int(ds.get("block_size", 5)) - 1)))
        depth = spec_tokens if requested_depth is None else requested_depth
        if depth <= 0:
            return 0, None
        return depth, DsparkDraftConfig(
            enabled=True,
            block_size=int(ds.get("block_size", spec_tokens + 1)),
            speculative_tokens=spec_tokens,
            confidence_enabled=bool(ds.get("confidence_enabled", False)),
        )

    depth = requested_depth if requested_depth is not None else 0
    if depth > 0 and enabled and method not in ("", "prompt_lookup"):
        log.warning(
            "speculation.method=%s has no CLI wiring yet "
            "(TD-SERVE-SPECULATION) — arming prompt-lookup only", method)
    return max(0, depth), None


# ---------------------------------------------------------------------------
# Stack construction
# ---------------------------------------------------------------------------

@dataclass
class ServeStack:
    """A fully-wired serve stack.  Owns the engine (stop via shutdown())."""
    engine: Any                       # the layerstorm_engine module
    info: Any                         # EngineInfo
    orchestrator: Orchestrator
    server: LayerStoRmServer
    tokenizer: Any
    options: ServeOptions

    def shutdown(self) -> None:
        """Tear down in reverse boot order: orchestrator → server →
        engine.  The orchestrator's request path is synchronous (every
        free_sequence is awaited in-line), so no settle drain is needed."""
        self.orchestrator.shutdown()
        self.server.shutdown()
        self.engine.stop_engine()


def _import_engine() -> Any:
    # NCCL SONAME collision: the engine module links the SYSTEM libnccl.so.2,
    # while torch ships its own under nvidia/nccl/lib. Only one can serve a
    # process — whichever loads first — so importing the engine first pins the
    # system copy, and a torch newer than it then fails to resolve a symbol it
    # needs (observed: `libtorch_cuda.so: undefined symbol: ncclCommResume`,
    # torch 2.13 + bundled NCCL 2.29.7 against system NCCL 2.29.3). Nothing
    # here imports torch, but the tokenizer does a few lines later
    # (transformers pulls it inside from_pretrained), so the collision would
    # hit at boot. Load torch FIRST when it is installed: its NCCL is the
    # newer of the two and the engine is backward-compatible with it.
    # Absent torch (no guided decoding) this is a no-op.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    try:
        import layerstorm_engine
    except ImportError as exc:
        raise RuntimeError(
            "layerstorm_engine pybind module not importable — build it "
            "first: cmake --build build --target layerstorm_engine "
            f"({exc})") from exc
    return layerstorm_engine


def build_stack(
    opts: ServeOptions,
    *,
    config: dict | None = None,
    engine_module: Any = None,
    tokenizer: Any = None,
    chat_template: Any = None,
) -> ServeStack:
    """Boot engine + loop + HTTP server; returns the wired stack.

    ``engine_module`` / ``tokenizer`` / ``chat_template`` are injectable
    for tests; production callers pass none of them.  On any failure after
    engine start, the engine is stopped before re-raising.
    """
    engine = engine_module if engine_module is not None else _import_engine()
    cfg = config if config is not None else load_config(opts.config_path)
    model_cfg = cfg.get("model") or {}
    weights_path = model_cfg.get("weights_path") or ""

    # ── Tokenizer + chat template (before engine start: fail fast) ──────
    sampling_defaults: dict = {}
    if tokenizer is None:
        tok_dir = resolve_tokenizer_dir(
            weights_path, opts.tokenizer_path, model_name=opts.model_name)
        log.info("loading tokenizer from %s", tok_dir)
        tokenizer = TokenizerWrapper(str(tok_dir))
        if chat_template is None:
            ct_dir = resolve_asset_dir(
                "chat_template.jinja", tok_dir, weights_path,
                model_name=opts.model_name, present=_has_chat_template)
            chat_template = ChatTemplateRenderer(ct_dir)
        gc_dir = resolve_asset_dir(
            "generation_config.json", tok_dir, weights_path,
            model_name=opts.model_name)
        sampling_defaults = read_sampling_defaults(
            gc_dir, opts.tokenizer_mode)
        if sampling_defaults:
            log.info("sampling defaults (%s, generation_config.json): %s",
                     opts.tokenizer_mode, sampling_defaults)
    elif chat_template is None:
        raise ValueError(
            "chat_template must be provided when tokenizer is injected")

    special = tokenizer.special_tokens
    if not special.eos_token_ids:
        log.warning("no EOS token ids detected — generations only stop at "
                    "max_tokens")
    # TD-VOCAB-AUTODETECT fallback chain (metadata display only): the
    # engine's resolved width (orch.metadata.vocab_size, below) wins; the
    # config dict is next; the tokenizer is the LAST resort.
    vocab_size = int(model_cfg.get("vocab_size") or 0)
    if vocab_size <= 0:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    model_type = model_cfg.get("architecture", "") or ""

    # Correctness gate (TD-SERVE-CONCURRENCY): the loop's Python-written
    # sideband command-input slots are single-owner — MULTIPLE concurrent
    # in-flight generations clobber each other's token/batch-descriptor
    # inputs (TD-ORCH-SIDEBAND-INPUT-MULTI).  Until that lands, the serve
    # stack admits ONE generation at a time; extra requests QUEUE at the
    # HTTP layer (bounded FIFO, serving.max_queued_requests) and only
    # queue overflow errors (503 + Retry-After).
    max_concurrent = opts.max_concurrent
    if max_concurrent > 1:
        log.warning(
            "clamping max_concurrent %d → 1: multi-request decode is not "
            "yet correct (TD-ORCH-SIDEBAND-INPUT-MULTI / "
            "TD-SERVE-CONCURRENCY)", max_concurrent)
        max_concurrent = 1

    # ── Engine boot (the long pole: weights + VRAM arenas + daemon) ─────
    log.info("starting engine (config=%s, test_engine=%s) ...",
             opts.config_path, opts.use_test_engine)
    orch = Orchestrator.boot(
        opts.config_path,
        engine_module=engine,
        eos_token_ids=tuple(special.eos_token_ids),
        speculation_depth=opts.speculation_depth,
        test_engine=opts.use_test_engine,
    )
    info = orch.info
    try:
        import dataclasses as _dc
        meta = _dc.replace(
            orch.metadata,
            think_start_token_id=special.think_start_token_id,
            think_end_token_id=special.think_end_token_id,
            vocab_size=(orch.metadata.vocab_size
                        if orch.metadata.vocab_size > 0 else vocab_size))
        orch.metadata = meta
        log.info("engine up: %d layers (%d MoE), %d experts, %d GPUs; "
                 "speculation=%s (gamma=%d, conf=%.2f), arm=%s%s",
                 meta.num_layers, meta.num_moe_layers, meta.num_experts,
                 meta.num_gpus, orch.spec.enabled, orch.spec.gamma,
                 orch.spec.conf_thresh, orch.bridge.route_arm,
                 " far" if orch.bridge.use_far else "")

        server = LayerStoRmServer(
            orchestrator=orch,
            tokenizer=tokenizer,
            chat_template=chat_template,
            metadata=meta,
            model_name=opts.model_name or "layerstorm",
            model_type=model_type,
            host=opts.host,
            port=opts.port,
            max_concurrent=max_concurrent,
            max_queued_requests=opts.max_queued_requests,
            sse_heartbeat_seconds=opts.sse_heartbeat_seconds,
            max_sequence_length=opts.max_sequence_length,
            tool_call_parser=opts.tool_call_parser,
            enable_auto_tool_choice=opts.enable_auto_tool_choice,
            reasoning_parser=opts.reasoning_parser,
            reasoning_config=_parse_reasoning_config(opts.reasoning_config),
            tokenizer_mode=opts.tokenizer_mode,
            sampling_defaults=sampling_defaults,
        )
    except Exception:
        engine.stop_engine()
        raise

    return ServeStack(
        engine=engine, info=info, orchestrator=orch,
        server=server, tokenizer=tokenizer, options=opts,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_stack(stack: ServeStack, *, install_signals: bool = True) -> None:
    """Start the HTTP server thread and run the orchestrator loop.

    Blocks the calling (main) thread inside OrchestratorLoop.run() until
    SIGINT/SIGTERM (or loop.shutdown() from another thread), then tears
    the stack down.  Signal handlers run between bytecodes of the main
    thread — i.e. inside the loop — so shutdown is always observed.
    """
    stack.server.run()
    if not stack.server.wait_ready(timeout=30.0):
        stack.shutdown()
        raise RuntimeError(
            f"HTTP server failed to start on "
            f"{stack.options.host}:{stack.options.port}")

    if install_signals:
        def _on_signal(signum: int, _frame: Any) -> None:
            log.info("received signal %d — shutting down", signum)
            stack.orchestrator.shutdown()
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

    log.info("serving '%s' at http://%s:%d/v1 (Ctrl-C to stop)",
             stack.options.model_name, stack.options.host,
             stack.server.bound_port or stack.options.port)
    try:
        stack.orchestrator.run()
    except KeyboardInterrupt:
        log.info("interrupted — shutting down")
    finally:
        stack.shutdown()
        log.info("serve stack stopped")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.model and not args.autoconfig:
        log.error("--model requires --autoconfig (a hand recipe boots via "
                  "--config)")
        return 2
    if (args.accuracy or args.prefer) and not args.autoconfig:
        log.error("--accuracy/--prefer are autoconfig levers — pass "
                  "--autoconfig (they change the derivation, not a served "
                  "recipe)")
        return 2
    if not args.config and not args.model:
        log.error("pass --config <recipe>, or --autoconfig --model "
                  "<weights> to derive one")
        return 2

    # P-31: serve.py --autoconfig turns the serving-shape flags into HARD
    # solver constraints (pins) — as ServeOptions overrides alone they
    # would change the HTTP surface but not the engine sizing, which reads
    # serving.* from the config file (the KV pool is sized for
    # max_concurrent_requests concurrent sequences).
    autoconfig_pins = None
    if args.autoconfig and (args.max_sequence_length is not None
                            or args.max_concurrent is not None):
        from autoconfig.pins import parse_pin_args
        pin_args = []
        if args.max_sequence_length is not None:
            pin_args.append(
                f"serving.max_sequence_length={args.max_sequence_length}")
        if args.max_concurrent is not None:
            pin_args.append(
                f"serving.max_concurrent_requests={args.max_concurrent}")
        autoconfig_pins = parse_pin_args(pin_args)

    derived_this_boot = False
    if args.autoconfig and args.model and not args.config:
        # derive-from-weights route: CPU-only, artifacts beside the
        # weights reused; the derived recipe is then served like any
        # --config file
        from autoconfig.pipeline import derive_for_serve
        rc, out = derive_for_serve(
            args.model, repo_root=".",
            prefer=args.prefer, accuracy=args.accuracy,
            pins=autoconfig_pins, redetect=args.autoconfig_redetect)
        if rc != 0:
            log.error("autoconfig --model derivation failed (exit %d) — "
                      "refusing to boot a guessed config", rc)
            return rc
        log.warning("autoconfig: derived %s from --model %s — serving it",
                    out, args.model)
        args.config = out
        args.autoconfig = False
        derived_this_boot = True  # derivation done; the enabled-in-config
                                  # branch below must not run a redundant
                                  # second derivation this boot

    try:
        config = load_config(args.config)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("cannot load config %s: %s", args.config, exc)
        return 1
    # Opt-in autoconfig (TD-AUTOCONFIG-HARDWARE-FIT): derive/reuse a
    # hardware-fit recipe and re-point BOTH consumers of the config path at
    # the persisted file (this dict AND the C++ start_engine(path) inside
    # Orchestrator.boot must see the same bytes). Default OFF — a hand-tuned
    # recipe always wins unless the flag or autoconfig.enabled asks.
    #
    # NEVER a silent substitution (TD-AUTOCONFIG-SERVE-SUBSTITUTES-THE-
    # CONFIG): when the derived recipe is a DIFFERENT file than --config,
    # serving it is either explicitly requested (--autoconfig, logged at
    # WARNING with both paths) or REFUSED — autoconfig.enabled=true inside
    # the config alone only re-derives/serves in place when output_path IS
    # the --config file (a derived recipe refreshing itself). Booting a
    # file other than the one on the command line silently invalidated
    # A/B bisect arms.
    if (args.autoconfig or (config.get("autoconfig") or {}).get("enabled")) \
            and not derived_this_boot:
        from autoconfig.cli import default_output_path, run as autoconfig_run
        out = ((config.get("autoconfig") or {}).get("output_path")
               or default_output_path(args.config))
        substitutes = os.path.realpath(out) != os.path.realpath(args.config)
        if substitutes and not args.autoconfig:
            log.error(
                "config %s carries autoconfig.enabled=true with output_path "
                "%s — refusing to boot a different file than --config "
                "(TD-AUTOCONFIG-SERVE-SUBSTITUTES-THE-CONFIG). Either pass "
                "--autoconfig to derive and serve the derived recipe, boot "
                "the derived file directly (--config %s), or remove/disable "
                "the config's autoconfig block to serve %s as-is.",
                args.config, out, out, args.config)
            return 2
        rc = autoconfig_run(args.config, out,
                            redetect=args.autoconfig_redetect,
                            prefer=args.prefer, accuracy=args.accuracy,
                            pins=autoconfig_pins)
        if rc != 0:
            log.error("autoconfig failed (exit %d) — refusing to boot a "
                      "guessed config", rc)
            return rc
        if substitutes:
            log.warning(
                "autoconfig: --config %s is the derivation SOURCE — serving "
                "the derived recipe %s instead (explicitly requested via "
                "--autoconfig)", args.config, out)
        else:
            log.info("autoconfig: serving derived recipe %s (== --config)",
                     out)
        args.config = out
        config = load_config(out)
    # DETERMINISM SUPERFLAGS (flag 1: compute.deterministic /
    # LS_DETERMINISTIC = run-to-run determinism; flag 2:
    # compute.reference_trajectory_identity /
    # LS_REFERENCE_TRAJECTORY_IDENTITY = reference-trajectory identity,
    # which IMPLIES flag 1): applied on the FINAL config (post-autoconfig
    # repoint), BEFORE the engine import inside build_stack — os.environ
    # writes putenv through to the C++ getenv sites, and the engine's own
    # registry apply then sees conforming values.  Refuses loudly (exit 2)
    # on env pins that contradict the requested mode — including requesting
    # flag 2 while suppressing flag 1 — instead of silently overriding.
    from orchestrator.determinism import (DeterminismConflictError,
                                          apply_deterministic_mode)
    try:
        apply_deterministic_mode(config)
    except DeterminismConflictError as exc:
        log.error("%s", exc)
        return 2
    opts = resolve_options(config, args)
    try:
        stack = build_stack(opts, config=config)
    except Exception as exc:  # noqa: BLE001 — boot errors become exit code
        log.error("failed to boot serve stack: %s", exc)
        return 1
    run_stack(stack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
