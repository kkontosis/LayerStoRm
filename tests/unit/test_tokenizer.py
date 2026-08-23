"""Tests for tokenizer wrapper and special token detection."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from tokenizer.tokenizer_wrapper import (
    SpecialTokenIds,
    TokenizerWrapper,
    _detect_eos_from_config,
    _scan_added_tokens_for_think,
    _search_vocab_for_think,
    detect_special_tokens,
)

_TEST_DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "test-data"
_DEEPSEEK = _TEST_DATA / "DeepSeek-V3.2"
_GLM5 = _TEST_DATA / "GLM-5"
_KIMI = _TEST_DATA / "Kimi-K2.5"


# ---------------------------------------------------------------------------
# SpecialTokenIds dataclass
# ---------------------------------------------------------------------------

class TestSpecialTokenIds:

    def test_defaults(self):
        s = SpecialTokenIds()
        assert s.eos_token_ids == ()
        assert s.think_start_token_id == -1
        assert s.think_end_token_id == -2

    def test_custom_values(self):
        s = SpecialTokenIds(
            eos_token_ids=(1,),
            think_start_token_id=128798,
            think_end_token_id=128799,
        )
        assert s.eos_token_ids == (1,)
        assert s.think_start_token_id == 128798
        assert s.think_end_token_id == 128799

    def test_frozen(self):
        s = SpecialTokenIds()
        with pytest.raises(AttributeError):
            s.eos_token_ids = (42,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _detect_eos_from_config
# ---------------------------------------------------------------------------

class TestDetectEos:

    def test_scalar(self):
        assert _detect_eos_from_config({"eos_token_id": 1}) == (1,)

    def test_array(self):
        result = _detect_eos_from_config(
            {"eos_token_id": [154820, 154827, 154829]},
        )
        assert result == (154820, 154827, 154829)

    def test_missing(self):
        assert _detect_eos_from_config({}) == ()

    def test_empty_array(self):
        assert _detect_eos_from_config({"eos_token_id": []}) == ()

    def test_float_coercion(self):
        assert _detect_eos_from_config({"eos_token_id": 1.0}) == (1,)


# ---------------------------------------------------------------------------
# _scan_added_tokens_for_think
# ---------------------------------------------------------------------------

class TestScanAddedTokens:

    def test_found(self):
        cfg = {
            "added_tokens_decoder": {
                "100": {"content": "<think>"},
                "101": {"content": "</think>"},
            },
        }
        assert _scan_added_tokens_for_think(cfg) == (100, 101)

    def test_not_found(self):
        cfg = {"added_tokens_decoder": {"100": {"content": "[EOS]"}}}
        assert _scan_added_tokens_for_think(cfg) == (-1, -2)

    def test_no_added_tokens(self):
        assert _scan_added_tokens_for_think({}) == (-1, -2)

    def test_non_dict_entry(self):
        cfg = {"added_tokens_decoder": {"100": "bad"}}
        assert _scan_added_tokens_for_think(cfg) == (-1, -2)


# ---------------------------------------------------------------------------
# detect_special_tokens — real test-data
# ---------------------------------------------------------------------------

class TestDetectSpecialTokens:

    def test_deepseek_v32(self):
        s = detect_special_tokens(_DEEPSEEK)
        assert s.eos_token_ids == (1,)
        assert s.think_start_token_id == 128798
        assert s.think_end_token_id == 128799

    def test_glm5(self):
        s = detect_special_tokens(_GLM5)
        assert s.eos_token_ids == (154820, 154827, 154829)
        assert s.think_start_token_id == -1
        assert s.think_end_token_id == -2

    def test_kimi_k25(self):
        s = detect_special_tokens(_KIMI)
        assert s.eos_token_ids == (163585,)
        assert s.think_start_token_id == 163606
        assert s.think_end_token_id == 163607


class TestDetectEdgeCases:

    def test_missing_directory(self, tmp_path):
        s = detect_special_tokens(tmp_path / "nonexistent")
        assert s == SpecialTokenIds()

    def test_empty_directory(self, tmp_path):
        s = detect_special_tokens(tmp_path)
        assert s.eos_token_ids == ()
        assert s.think_start_token_id == -1

    def test_malformed_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{bad json")
        s = detect_special_tokens(tmp_path)
        assert s.eos_token_ids == ()

    def test_config_without_eos(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model_type": "test"}')
        s = detect_special_tokens(tmp_path)
        assert s.eos_token_ids == ()

    def test_only_tokenizer_config(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "added_tokens_decoder": {
                "50": {"content": "<think>"},
                "51": {"content": "</think>"},
            },
        }))
        s = detect_special_tokens(tmp_path)
        assert s.eos_token_ids == ()
        assert s.think_start_token_id == 50
        assert s.think_end_token_id == 51


# ---------------------------------------------------------------------------
# _search_vocab_for_think
# ---------------------------------------------------------------------------

class TestSearchVocabForThink:

    def _mock_tokenizer(self, vocab: dict[str, int], unk_id: int = -1):
        tok = MagicMock()
        tok.unk_token_id = unk_id
        tok.convert_tokens_to_ids = lambda t: vocab.get(t, unk_id)
        return tok

    def test_standard_think_tokens(self):
        tok = self._mock_tokenizer({"<think>": 100, "</think>": 101})
        assert _search_vocab_for_think(tok) == (100, 101)

    def test_deepseek_fullwidth_variants(self):
        tok = self._mock_tokenizer({
            "<｜thinking▁start｜>": 128798,
            "<｜thinking▁end｜>": 128799,
        })
        assert _search_vocab_for_think(tok) == (128798, 128799)

    def test_not_found(self):
        tok = self._mock_tokenizer({})
        assert _search_vocab_for_think(tok) == (-1, -2)

    def test_no_convert_method(self):
        tok = object()
        assert _search_vocab_for_think(tok) == (-1, -2)

    def test_standard_takes_priority(self):
        tok = self._mock_tokenizer({
            "<think>": 50,
            "<｜thinking▁start｜>": 999,
            "</think>": 51,
            "<｜thinking▁end｜>": 998,
        })
        assert _search_vocab_for_think(tok) == (50, 51)


# ---------------------------------------------------------------------------
# TokenizerWrapper
# ---------------------------------------------------------------------------

class TestTokenizerWrapperInit:

    def test_auto_requires_model_path(self):
        with pytest.raises(ValueError, match="no model_path"):
            TokenizerWrapper._resolve_path("auto", None)

    def test_explicit_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TokenizerWrapper._resolve_path(
                str(tmp_path / "nonexistent"), None,
            )

    def test_auto_resolves_to_model_path(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{}")
        p = TokenizerWrapper._resolve_path("auto", str(tmp_path))
        assert p == tmp_path

    def test_explicit_path_resolved(self, tmp_path):
        p = TokenizerWrapper._resolve_path(str(tmp_path), None)
        assert p == tmp_path

    @patch("tokenizer.tokenizer_wrapper.AutoTokenizer", create=True)
    def test_full_init_mocked(self, mock_auto_cls, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"eos_token_id": 42, "model_type": "test"}),
        )
        (tmp_path / "tokenizer_config.json").write_text("{}")

        mock_tok = MagicMock()
        mock_tok.unk_token_id = -1
        mock_tok.convert_tokens_to_ids = lambda t: -1

        with patch(
            "tokenizer.tokenizer_wrapper.AutoTokenizer",
            create=True,
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tok
            # Patch the import inside __init__
            import tokenizer.tokenizer_wrapper as tw
            orig = tw.__dict__.get("AutoTokenizer")
            try:
                tw.AutoTokenizer = mock_auto  # type: ignore[attr-defined]
                # We need to patch the local import, so patch builtins
                import importlib
                with patch.dict(
                    "sys.modules",
                    {"transformers": MagicMock(AutoTokenizer=mock_auto)},
                ):
                    wrapper = TokenizerWrapper(
                        tokenizer_path="auto",
                        model_path=str(tmp_path),
                    )
            finally:
                if orig is not None:
                    tw.AutoTokenizer = orig  # type: ignore[attr-defined]

        assert wrapper.eos_token_ids == (42,)


class TestTokenizerWrapperEncodeDecode:

    def _make_wrapper(self, mock_tok, tmp_path):
        (tmp_path / "config.json").write_text('{"eos_token_id": 1}')
        (tmp_path / "tokenizer_config.json").write_text("{}")
        mock_tok.unk_token_id = -1
        mock_tok.convert_tokens_to_ids = lambda t: -1
        mock_tok.__len__ = lambda self: 129280

        with patch.dict(
            "sys.modules",
            {"transformers": MagicMock(AutoTokenizer=MagicMock(
                from_pretrained=MagicMock(return_value=mock_tok),
            ))},
        ):
            return TokenizerWrapper(tokenizer_path=str(tmp_path))

    def test_encode(self, tmp_path):
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10, 20, 30]
        wrapper = self._make_wrapper(mock_tok, tmp_path)
        result = wrapper.encode("hello world")
        mock_tok.encode.assert_called_once_with(
            "hello world", add_special_tokens=False,
        )
        assert result == [10, 20, 30]

    def test_decode(self, tmp_path):
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello world"
        wrapper = self._make_wrapper(mock_tok, tmp_path)
        result = wrapper.decode([10, 20, 30])
        mock_tok.decode.assert_called_once_with([10, 20, 30])
        assert result == "hello world"

    def test_vocab_size(self, tmp_path):
        mock_tok = MagicMock()
        mock_tok.__len__ = lambda self: 129280
        wrapper = self._make_wrapper(mock_tok, tmp_path)
        assert wrapper.vocab_size == 129280

    def test_special_tokens_property(self, tmp_path):
        mock_tok = MagicMock()
        wrapper = self._make_wrapper(mock_tok, tmp_path)
        st = wrapper.special_tokens
        assert isinstance(st, SpecialTokenIds)
        assert st.eos_token_ids == (1,)
