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

log = logging.getLogger("layerstorm.serve")

# Files whose presence marks a usable HuggingFace tokenizer directory.
_TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer_config.json",
                      "tokenizer.model")


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
    p.add_argument("--config", required=True,
                   help="engine config JSON (config/schema.json)")
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
                   choices=["auto", "hf", "deepseek_v4"],
                   help="serving tokenizer mode (vLLM parity subset): "
                        "'auto' resolves from model.architecture "
                        "(deepseek_v4 models → deepseek_v4), 'hf' forces "
                        "the legacy behavior, 'deepseek_v4' switches "
                        "chat-template kwarg normalization + thinking "
                        "defaults to the DeepSeek-V4 rules "
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
    return p


def _resolve_tokenizer_mode(mode: str, architecture: str) -> str:
    """vLLM tokenizer-mode auto-default rule (ref/vllm
    vllm/config/model.py:617): "auto" resolves to "deepseek_v4" for
    deepseek_v4 models, else the legacy behavior ("hf")."""
    if mode == "auto":
        return "deepseek_v4" if architecture == "deepseek_v4" else "hf"
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
) -> Path:
    """Resolve the HuggingFace tokenizer directory for a model.

    Explicit ``tokenizer_path`` (anything but "auto") wins.  "auto" looks
    for tokenizer files in the weights directory (or, for single-file
    weights such as GGUF, in the file's parent directory).
    """
    if tokenizer_path and tokenizer_path != "auto":
        p = Path(tokenizer_path)
        if not p.exists():
            raise FileNotFoundError(
                f"serving.tokenizer_path does not exist: {p}")
        return p

    w = Path(weights_path)
    candidate = w if w.is_dir() else w.parent
    if any((candidate / m).is_file() for m in _TOKENIZER_MARKERS):
        return candidate
    raise FileNotFoundError(
        f"no HuggingFace tokenizer files ({', '.join(_TOKENIZER_MARKERS)}) "
        f"found in {candidate} — set serving.tokenizer_path in the config "
        "(GGUF-embedded tokenizers are not extracted; "
        "see TD-SERVE-GGUF-TOKENIZER)")


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
    if tokenizer is None:
        tok_dir = resolve_tokenizer_dir(weights_path, opts.tokenizer_path)
        log.info("loading tokenizer from %s", tok_dir)
        tokenizer = TokenizerWrapper(str(tok_dir))
        if chat_template is None:
            chat_template = ChatTemplateRenderer(tok_dir)
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
    # stack admits ONE generation at a time; extra requests get 429 +
    # Retry-After from the HTTP layer.
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
            max_sequence_length=opts.max_sequence_length,
            tool_call_parser=opts.tool_call_parser,
            enable_auto_tool_choice=opts.enable_auto_tool_choice,
            reasoning_parser=opts.reasoning_parser,
            reasoning_config=_parse_reasoning_config(opts.reasoning_config),
            tokenizer_mode=opts.tokenizer_mode,
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
    try:
        config = load_config(args.config)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("cannot load config %s: %s", args.config, exc)
        return 1
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
