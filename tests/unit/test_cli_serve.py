"""Tests for the CLI serve entrypoint (#71) — python/cli/serve.py.

Coverage:
  - argument parsing + serving-section/CLI override resolution
  - tokenizer directory auto-resolution (dir / single-file / explicit / miss)
  - speculation derivation (greedy default, explicit depth, dspark config)
  - build_stack failure path (engine stopped on post-start wiring error)
  - FULL-STACK smoke over the REAL pybind engine with null backends:
    HTTP → LayerStoRmServer → Orchestrator (bridge) → SPSC rings → C++ daemon
    → completion → HTTP response.  Null backends carry no weights, so the
    round trip deterministically finalizes with finish_reason "error" —
    the assertion is the LIVE end-to-end plumbing, not token quality
    (token correctness is tests/integration/test_serve_e2e.py, GPU-gated).
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from cli.serve import (
    ServeOptions,
    build_arg_parser,
    build_stack,
    derive_speculation,
    load_config,
    resolve_options,
    resolve_tokenizer_dir,
)
from orchestrator.dspark_draft import DsparkDraftConfig
from tokenizer.tokenizer_wrapper import SpecialTokenIds

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_TEST_CONFIG = str(
    _PROJECT_ROOT / "test-data" / "config" / "valid_deepseek_v3_2.json")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgParsing:

    def test_config_required(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args([])

    def test_defaults(self):
        args = build_arg_parser().parse_args(["--config", "c.json"])
        assert args.config == "c.json"
        assert args.host is None
        assert args.port is None
        assert args.speculation_depth is None
        assert args.log_level == "info"
        assert args.test_engine is False

    def test_overrides(self):
        args = build_arg_parser().parse_args([
            "--config", "c.json", "--host", "127.0.0.1", "--port", "9000",
            "--model-name", "m", "--tokenizer-path", "/tok",
            "--max-concurrent", "4", "--max-sequence-length", "1024",
            "--speculation-depth", "3", "--test-engine",
        ])
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.model_name == "m"
        assert args.tokenizer_path == "/tok"
        assert args.max_concurrent == 4
        assert args.max_sequence_length == 1024
        assert args.speculation_depth == 3
        assert args.test_engine is True


# ---------------------------------------------------------------------------
# Option resolution (config serving section + CLI overrides)
# ---------------------------------------------------------------------------

class TestResolveOptions:

    def _args(self, extra: list[str] | None = None):
        return build_arg_parser().parse_args(
            ["--config", "c.json"] + (extra or []))

    def test_serving_section_used(self):
        cfg = {
            "model": {"weights_path": "/models/foo/weights.gguf"},
            "serving": {"host": "10.0.0.1", "port": 8123,
                        "max_concurrent_requests": 7,
                        "max_sequence_length": 4096,
                        "tokenizer_path": "/tok"},
        }
        opts = resolve_options(cfg, self._args())
        assert opts.host == "10.0.0.1"
        assert opts.port == 8123
        assert opts.max_concurrent == 7
        assert opts.max_sequence_length == 4096
        assert opts.tokenizer_path == "/tok"
        # model_name derives from the weights file stem
        assert opts.model_name == "weights"

    def test_cli_overrides_win(self):
        cfg = {"serving": {"host": "10.0.0.1", "port": 8123}}
        opts = resolve_options(cfg, self._args(
            ["--host", "0.0.0.0", "--port", "9999", "--model-name", "mm"]))
        assert opts.host == "0.0.0.0"
        assert opts.port == 9999
        assert opts.model_name == "mm"

    def test_defaults_without_serving_section(self):
        opts = resolve_options({}, self._args())
        assert opts.host == "0.0.0.0"
        assert opts.port == 8000
        assert opts.max_concurrent == 32
        assert opts.max_sequence_length == 32768
        assert opts.tokenizer_path == "auto"
        assert opts.model_name == "layerstorm"

    def test_model_name_from_directory_weights(self):
        cfg = {"model": {"weights_path": "/models/DeepSeek-V3.2"}}
        opts = resolve_options(cfg, self._args())
        assert opts.model_name == "DeepSeek-V3.2"


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------

class TestResolveTokenizerDir:

    def test_explicit_path_wins(self, tmp_path):
        tok = tmp_path / "tok"
        tok.mkdir()
        assert resolve_tokenizer_dir("/nonexistent", str(tok)) == tok

    def test_explicit_path_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="tokenizer_path"):
            resolve_tokenizer_dir("/w", "/definitely/not/here")

    def test_auto_weights_dir(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        assert resolve_tokenizer_dir(tmp_path, "auto") == tmp_path

    def test_auto_single_file_weights_uses_parent(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{}")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        assert resolve_tokenizer_dir(gguf, "auto") == tmp_path

    def test_auto_miss_raises_with_guidance(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        with pytest.raises(FileNotFoundError,
                           match="serving.tokenizer_path"):
            resolve_tokenizer_dir(gguf, "auto")


# ---------------------------------------------------------------------------
# Speculation derivation
# ---------------------------------------------------------------------------

class TestDeriveSpeculation:

    def test_default_is_greedy(self):
        depth, ds = derive_speculation({}, None)
        assert depth == 0
        assert ds is None

    def test_explicit_depth_arms_prompt_lookup(self):
        depth, ds = derive_speculation({}, 4)
        assert depth == 4
        assert ds is None

    def test_dspark_config_derives_depth_and_planner(self):
        cfg = {"speculation": {"enabled": True, "method": "dspark",
                               "dspark": {"block_size": 8,
                                          "speculative_tokens": 7,
                                          "confidence_enabled": True}}}
        depth, ds = derive_speculation(cfg, None)
        assert depth == 7
        assert isinstance(ds, DsparkDraftConfig)
        assert ds.enabled is True
        assert ds.block_size == 8
        assert ds.speculative_tokens == 7
        assert ds.confidence_enabled is True

    def test_dspark_explicit_zero_forces_greedy(self):
        cfg = {"speculation": {"enabled": True, "method": "dspark",
                               "dspark": {"block_size": 8,
                                          "speculative_tokens": 7}}}
        depth, ds = derive_speculation(cfg, 0)
        assert depth == 0
        assert ds is None

    def test_disabled_speculation_ignored(self):
        cfg = {"speculation": {"enabled": False, "method": "dspark",
                               "dspark": {"speculative_tokens": 7}}}
        depth, ds = derive_speculation(cfg, None)
        assert depth == 0
        assert ds is None

    def test_unwired_method_falls_back_to_prompt_lookup(self):
        cfg = {"speculation": {"enabled": True, "method": "mtp"}}
        depth, ds = derive_speculation(cfg, 2)
        assert depth == 2
        assert ds is None


# ---------------------------------------------------------------------------
# build_stack helpers
# ---------------------------------------------------------------------------

def _mock_tokenizer(eos: tuple[int, ...] = (1,)) -> MagicMock:
    tok = MagicMock()
    tok.special_tokens = SpecialTokenIds(eos_token_ids=eos)
    tok.eos_token_ids = eos
    tok.vocab_size = 32000
    tok.encode.return_value = [5, 6, 7]
    tok.decode.return_value = "mocked"
    return tok


class TestBuildStackFailurePath:

    def test_engine_stopped_on_wiring_error(self):
        """A failure after start_engine must stop the engine (no leak)."""
        engine = MagicMock()
        engine.start_engine_test.return_value = MagicMock()
        engine.query_buffer_ids.side_effect = RuntimeError("boom")
        opts = ServeOptions(config_path=_TEST_CONFIG, use_test_engine=True)
        with pytest.raises(RuntimeError, match="boom"):
            build_stack(
                opts,
                config=load_config(_TEST_CONFIG),
                engine_module=engine,
                tokenizer=_mock_tokenizer(),
                chat_template=MagicMock(),
            )
        engine.stop_engine.assert_called_once()

    def test_injected_tokenizer_requires_chat_template(self):
        opts = ServeOptions(config_path=_TEST_CONFIG, use_test_engine=True)
        with pytest.raises(ValueError, match="chat_template"):
            build_stack(opts, config={}, engine_module=MagicMock(),
                        tokenizer=_mock_tokenizer())


# ---------------------------------------------------------------------------
# FULL-STACK smoke: real pybind engine (null backends) + HTTP + loop thread
# ---------------------------------------------------------------------------

class TestServeStackNullEngine:
    """The serve stack against the REAL daemon with null backends.

    Placed last in this module (heaviest test here).  No CUDA/GPU needed.
    """

    def test_full_stack_round_trip(self):
        import layerstorm_engine

        opts = ServeOptions(
            config_path=_TEST_CONFIG,
            model_name="null-model",
            use_test_engine=True,
            cycle_budget_us=0.0,
            max_concurrent=8,   # must clamp to 1 (TD-SERVE-CONCURRENCY)
        )
        tok = _mock_tokenizer()
        stack = build_stack(
            opts,
            config=load_config(_TEST_CONFIG),
            engine_module=layerstorm_engine,
            tokenizer=tok,
            chat_template=MagicMock(),
        )
        loop_thread = threading.Thread(target=stack.orchestrator.run, daemon=True)
        try:
            # Metadata really came from the live engine.
            assert stack.orchestrator.metadata.num_layers == stack.info.num_layers
            # Tokenizer-derived fields were injected into loop metadata.
            assert stack.orchestrator.metadata.eos_token_ids == (1,)
            assert stack.orchestrator.metadata.vocab_size > 0
            # Concurrency clamp (TD-SERVE-CONCURRENCY): multi-request
            # decode is not correct yet (TD-ORCH-SIDEBAND-INPUT-MULTI) —
            # the serve stack must fail closed to 1 in-flight generation.
            assert stack.server._max_concurrent == 1

            loop_thread.start()
            client = TestClient(stack.server.app)

            resp = client.get("/health")
            assert resp.status_code == 200

            resp = client.get("/v1/models")
            assert resp.status_code == 200
            assert resp.json()["data"][0]["id"] == "null-model"

            # Live round trip: HTTP → loop → rings → C++ daemon →
            # completion → HTTP.  Null backends carry no layer weights, so
            # the daemon deterministically answers the first RUN_ATTENTION
            # with CMP_ERROR and the request fails — the ROUND TRIP is the
            # assertion, and (TD-SERVE-ERROR-MASKING) the failure must
            # reach the client as an OpenAI error with a 500-class status,
            # carrying the ENGINE's detail end-to-end.  It used to be a
            # 200 with an empty completion.
            resp = client.post("/v1/completions", json={
                "model": "null-model",
                "prompt": [5, 6, 7],
                "max_tokens": 2,
            })
            assert resp.status_code == 500
            body = resp.json()
            assert "choices" not in body
            err = body["error"]
            assert err["type"] == "server_error"
            assert err["code"] == "internal_error"
            # The engine's own attention failure must surface (INV-SERVE-
            # ERROR).  Message text differs by prefill path: chunk-64 FAR
            # says "attention dispatch failed"; the mini-superchunk path
            # (GLM default since 2026-08-23) surfaces the raw
            # "CMP_ERROR (attn L<k>)" completion.
            assert ("attention dispatch failed" in err["message"]
                    or "attn L" in err["message"])
        finally:
            stack.orchestrator.shutdown()
            loop_thread.join(timeout=10.0)
            # stack.shutdown() re-runs cycles safely after the thread exits
            # and stops the engine.
            stack.shutdown()
        assert not loop_thread.is_alive()


# ---------------------------------------------------------------------------
# vLLM-parity parser flags (--tool-call-parser / --enable-auto-tool-choice /
# --reasoning-parser) — CLI overrides the config serving section.
# ---------------------------------------------------------------------------

class TestParserFlagResolution:

    @staticmethod
    def _args(extra=()):
        return build_arg_parser().parse_args(
            ["--config", "c.json", *extra])

    def test_defaults_disabled(self):
        opts = resolve_options({}, self._args())
        assert opts.tool_call_parser == ""
        assert opts.enable_auto_tool_choice is False
        assert opts.reasoning_parser == ""

    def test_config_serving_section_used(self):
        cfg = {"serving": {
            "tool_call_parser": "glm47",
            "enable_auto_tool_choice": True,
            "reasoning_parser": "glm45",
        }}
        opts = resolve_options(cfg, self._args())
        assert opts.tool_call_parser == "glm47"
        assert opts.enable_auto_tool_choice is True
        assert opts.reasoning_parser == "glm45"

    def test_cli_overrides_config(self):
        cfg = {"serving": {"tool_call_parser": "", "reasoning_parser": ""}}
        opts = resolve_options(cfg, self._args([
            "--tool-call-parser", "glm47",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "glm45",
        ]))
        assert opts.tool_call_parser == "glm47"
        assert opts.enable_auto_tool_choice is True
        assert opts.reasoning_parser == "glm45"

    def test_enable_flag_absent_keeps_config_value(self):
        cfg = {"serving": {"tool_call_parser": "glm47",
                           "enable_auto_tool_choice": True}}
        opts = resolve_options(cfg, self._args())
        assert opts.enable_auto_tool_choice is True


class TestDeepSeekV4ServingArgs:
    """--tokenizer-mode / --reasoning-config resolution (ticket I)."""

    @staticmethod
    def _args(extra=()):
        return build_arg_parser().parse_args(
            ["--config", "c.json", *extra])

    def test_tokenizer_mode_auto_resolves_hf_by_default(self):
        opts = resolve_options({}, self._args())
        assert opts.tokenizer_mode == "hf"
        assert opts.reasoning_config == ""

    def test_tokenizer_mode_auto_resolves_deepseek_v4_from_arch(self):
        cfg = {"model": {"architecture": "deepseek_v4"}}
        opts = resolve_options(cfg, self._args())
        assert opts.tokenizer_mode == "deepseek_v4"

    def test_tokenizer_mode_explicit_hf_wins_over_arch(self):
        cfg = {"model": {"architecture": "deepseek_v4"}}
        opts = resolve_options(cfg, self._args(["--tokenizer-mode", "hf"]))
        assert opts.tokenizer_mode == "hf"

    def test_tokenizer_mode_config_serving_section(self):
        cfg = {"model": {"architecture": "glm_moe_dsa"},
               "serving": {"tokenizer_mode": "deepseek_v4"}}
        opts = resolve_options(cfg, self._args())
        assert opts.tokenizer_mode == "deepseek_v4"

    def test_reasoning_config_cli_and_serving(self):
        cfg = {"serving": {"reasoning_config":
                           '{"reasoning_end_str": "</r>"}'}}
        opts = resolve_options(cfg, self._args())
        assert opts.reasoning_config == '{"reasoning_end_str": "</r>"}'
        opts = resolve_options(cfg, self._args(
            ["--reasoning-config", '{"reasoning_start_str": ""}']))
        assert opts.reasoning_config == '{"reasoning_start_str": ""}'

    def test_parse_reasoning_config(self):
        from cli.serve import _parse_reasoning_config
        assert _parse_reasoning_config("") == {}
        assert _parse_reasoning_config(
            '{"reasoning_start_str": "", "reasoning_end_str": "</think>"}'
        ) == {"reasoning_start_str": "", "reasoning_end_str": "</think>"}
        with pytest.raises(ValueError):
            _parse_reasoning_config('["not", "an", "object"]')

    def test_invalid_tokenizer_mode_rejected(self):
        with pytest.raises(SystemExit):
            self._args(["--tokenizer-mode", "bogus"])
