"""P-34 step 1: ONE tokenizer-resolution precedence, everywhere.

Pins the shared search (tokenizer.locator) and both consumers (serve's
resolve_tokenizer_dir, autoconfig's modelprobe): explicit path always wins;
then the WEIGHTS directory; then a name-prefix sibling beside the weights;
repo test-data/ is a FALLBACK ONLY and must be surfaced loudly (a WARNING
log naming the resolved path).  Negative controls prove the weights dir
wins even when a same-named test-data dir exists.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python"))

from tokenizer.locator import (  # noqa: E402
    TIER_SIBLING, TIER_TEST_DATA, TIER_WEIGHTS, locate_tokenizer_dir)
from autoconfig import modelprobe  # noqa: E402
from cli.serve import resolve_tokenizer_dir  # noqa: E402


def _mk_repo(tmp_path: Path, name: str = "GLM-9.9-Test") -> Path:
    """A fake repo root with a matching test-data tokenizer dir."""
    td = tmp_path / "repo" / "test-data" / name
    td.mkdir(parents=True)
    (td / "tokenizer.json").write_text("{}")
    return tmp_path / "repo"


def _mk_weights(tmp_path: Path, with_tokenizer: bool) -> Path:
    w = tmp_path / "srv" / "GLM-9.9-Test-GGUF" / "Q4"
    w.mkdir(parents=True)
    (w / "model-00001-of-00001.gguf").write_bytes(b"GGUF")
    if with_tokenizer:
        (w / "tokenizer.json").write_text("{}")
    return w


class TestLocator:
    def test_weights_dir_wins_over_same_named_test_data(self, tmp_path):
        # NEGATIVE CONTROL: test-data candidate exists AND matches the
        # model name — the weights dir must still win.
        repo = _mk_repo(tmp_path)
        w = _mk_weights(tmp_path, with_tokenizer=True)
        d, tier = locate_tokenizer_dir(
            str(w), model_name="glm-9.9-test", repo_root=str(repo))
        assert d == str(w)
        assert tier == TIER_WEIGHTS

    def test_single_file_weights_use_parent(self, tmp_path):
        w = _mk_weights(tmp_path, with_tokenizer=True)
        d, tier = locate_tokenizer_dir(str(w / "model-00001-of-00001.gguf"))
        assert (d, tier) == (str(w), TIER_WEIGHTS)

    def test_sibling_prefix_beats_test_data(self, tmp_path):
        repo = _mk_repo(tmp_path)
        w = tmp_path / "srv" / "GLM-9.9-Test-GGUF-Q4_K_XL"
        w.mkdir(parents=True)
        (w / "a.gguf").write_bytes(b"GGUF")
        sib = tmp_path / "srv" / "GLM-9.9-Test"
        sib.mkdir()
        (sib / "tokenizer.json").write_text("{}")
        d, tier = locate_tokenizer_dir(
            str(w), model_name="glm-9.9-test", repo_root=str(repo))
        assert (d, tier) == (str(sib), TIER_SIBLING)

    def test_test_data_is_the_last_resort(self, tmp_path):
        repo = _mk_repo(tmp_path)
        w = _mk_weights(tmp_path, with_tokenizer=False)
        d, tier = locate_tokenizer_dir(
            str(w), model_name="glm-9.9-test", repo_root=str(repo))
        assert d == str(repo / "test-data" / "GLM-9.9-Test")
        assert tier == TIER_TEST_DATA

    def test_nothing_found_is_empty(self, tmp_path):
        w = _mk_weights(tmp_path, with_tokenizer=False)
        assert locate_tokenizer_dir(str(w)) == ("", "")

    def test_test_data_needs_both_root_and_name(self, tmp_path):
        repo = _mk_repo(tmp_path)
        w = _mk_weights(tmp_path, with_tokenizer=False)
        assert locate_tokenizer_dir(str(w), repo_root=str(repo)) == ("", "")
        assert locate_tokenizer_dir(str(w), model_name="glm-9.9-test") == \
            ("", "")


class TestServeResolution:
    def test_explicit_always_wins(self, tmp_path):
        # NEGATIVE CONTROL: weights dir has a tokenizer too.
        w = _mk_weights(tmp_path, with_tokenizer=True)
        tok = tmp_path / "elsewhere"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}")
        assert resolve_tokenizer_dir(w, str(tok)) == tok

    def test_weights_dir_wins_even_with_test_data_present(self, tmp_path):
        repo = _mk_repo(tmp_path)
        w = _mk_weights(tmp_path, with_tokenizer=True)
        assert resolve_tokenizer_dir(
            w, "auto", model_name="glm-9.9-test",
            repo_root=repo) == w

    def test_test_data_fallback_logs_loudly(self, tmp_path, caplog):
        repo = _mk_repo(tmp_path)
        w = _mk_weights(tmp_path, with_tokenizer=False)
        with caplog.at_level(logging.WARNING, logger="layerstorm.serve"):
            d = resolve_tokenizer_dir(
                w, "auto", model_name="glm-9.9-test", repo_root=repo)
        assert d == repo / "test-data" / "GLM-9.9-Test"
        warn = [r for r in caplog.records
                if "TOKENIZER FALLBACK" in r.getMessage()]
        assert warn, "test-data fallback must log a WARNING"
        assert str(d) in warn[0].getMessage()

    def test_absent_everywhere_raises(self, tmp_path):
        w = _mk_weights(tmp_path, with_tokenizer=False)
        with pytest.raises(FileNotFoundError, match="tokenizer_path"):
            resolve_tokenizer_dir(w, "auto", model_name="nope",
                                  repo_root=tmp_path)


class TestModelprobeFallbackIsLoud:
    def test_probe_warns_and_marks_provenance_on_test_data(
            self, tmp_path, caplog):
        # display name falls back to the weights dir basename ("m-GGUF"),
        # so the matching test-data dir carries the same name.
        repo = _mk_repo(tmp_path, "m-GGUF")
        w = tmp_path / "repo" / "m-GGUF"
        w.mkdir()
        import struct
        # minimal GGUF v3 header with 0 tensors / 0 kv
        (w / "m-00001-of-00001.gguf").write_bytes(
            b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
        with caplog.at_level(logging.WARNING, logger="layerstorm.autoconfig"):
            try:
                src = modelprobe.probe_model(
                    str(w), str(repo),
                    base_model_section={"architecture": "glm5_next"})
            except Exception:
                pytest.skip("probe needs fuller GGUF metadata on this path")
        assert src.tokenizer_path == "test-data/m-GGUF"
        assert any("TOKENIZER FALLBACK" in r.getMessage()
                   for r in caplog.records)
        assert any("FALLBACK from repo test-data/" in p
                   for p in src.provenance)

    def test_probe_quiet_when_tokenizer_beside_weights(
            self, tmp_path, caplog):
        repo = _mk_repo(tmp_path, "m-GGUF")
        w = tmp_path / "repo" / "m-GGUF"
        w.mkdir()
        import struct
        (w / "m-00001-of-00001.gguf").write_bytes(
            b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
        (w / "tokenizer.json").write_text("{}")
        with caplog.at_level(logging.WARNING, logger="layerstorm.autoconfig"):
            try:
                src = modelprobe.probe_model(
                    str(w), str(repo),
                    base_model_section={"architecture": "glm5_next"})
            except Exception:
                pytest.skip("probe needs fuller GGUF metadata on this path")
        assert src.tokenizer_path == "m-GGUF"
        assert not any("TOKENIZER FALLBACK" in r.getMessage()
                       for r in caplog.records)


class TestAssetResolution:
    """P-34 widened scope: chat_template.jinja / generation_config.json /
    config.json follow the SAME precedence as the tokenizer."""

    def test_asset_beside_weights_wins_over_test_data(self, tmp_path):
        # NEGATIVE CONTROL for a non-tokenizer asset.
        from tokenizer.locator import locate_model_asset
        repo = _mk_repo(tmp_path)
        (repo / "test-data" / "GLM-9.9-Test" /
         "generation_config.json").write_text("{}")
        w = _mk_weights(tmp_path, with_tokenizer=True)
        (w / "generation_config.json").write_text("{}")
        f, tier = locate_model_asset(
            str(w), "generation_config.json",
            model_name="glm-9.9-test", repo_root=str(repo))
        assert (f, tier) == (str(w / "generation_config.json"), TIER_WEIGHTS)

    def test_asset_falls_back_to_test_data_and_serve_logs(
            self, tmp_path, caplog):
        from cli.serve import resolve_asset_dir
        repo = _mk_repo(tmp_path)
        (repo / "test-data" / "GLM-9.9-Test" /
         "generation_config.json").write_text("{}")
        w = _mk_weights(tmp_path, with_tokenizer=True)
        with caplog.at_level(logging.WARNING, logger="layerstorm.serve"):
            d = resolve_asset_dir(
                "generation_config.json", w, w,
                model_name="glm-9.9-test", repo_root=repo)
        assert d == repo / "test-data" / "GLM-9.9-Test"
        assert any("METADATA FALLBACK" in r.getMessage()
                   for r in caplog.records)

    def test_explicit_tokenizer_dir_keeps_its_assets(self, tmp_path):
        # explicit tokenizer_path dir carries the asset -> no search,
        # even when a copy exists beside the weights.
        from cli.serve import resolve_asset_dir
        w = _mk_weights(tmp_path, with_tokenizer=True)
        (w / "generation_config.json").write_text("{}")
        tok = tmp_path / "explicit"
        tok.mkdir()
        (tok / "generation_config.json").write_text("{}")
        d = resolve_asset_dir("generation_config.json", tok, w,
                              model_name="x", repo_root=tmp_path)
        assert d == tok

    def test_absent_everywhere_returns_tok_dir(self, tmp_path):
        from cli.serve import resolve_asset_dir
        w = _mk_weights(tmp_path, with_tokenizer=True)
        assert resolve_asset_dir("generation_config.json", w, w,
                                 model_name="x", repo_root=tmp_path) == w

    def test_embedded_chat_template_suppresses_search(self, tmp_path):
        from cli.serve import _has_chat_template, resolve_asset_dir
        repo = _mk_repo(tmp_path)
        (repo / "test-data" / "GLM-9.9-Test" /
         "chat_template.jinja").write_text("x")
        w = _mk_weights(tmp_path, with_tokenizer=True)
        (w / "tokenizer_config.json").write_text(
            '{"chat_template": "{{ messages }}"}')
        d = resolve_asset_dir("chat_template.jinja", w, w,
                              model_name="glm-9.9-test", repo_root=repo,
                              present=_has_chat_template)
        assert d == w
