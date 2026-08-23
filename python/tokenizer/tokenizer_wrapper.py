"""HuggingFace tokenizer wrapper with special token auto-detection.

Two entry points:
  detect_special_tokens(model_path) — stdlib-only, reads JSON config files.
  TokenizerWrapper(path)            — full HuggingFace tokenizer (encode/decode).

Requires ``transformers`` for TokenizerWrapper; detect_special_tokens uses
only stdlib (json, pathlib, logging).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_KNOWN_THINK_TOKENS: dict[str, tuple[int, int]] = {
    "deepseek_v32": (128798, 128799),
    "deepseek_v3": (128798, 128799),
    # DeepSeek-V4-Flash respecialized the token region ≥128793 (GGUF
    # header census 2026-08-20): <think>=128821, </think>=128822.
    "deepseek_v4": (128821, 128822),
}

_THINK_START_CANDIDATES = ("<think>", "<｜thinking▁start｜>")
_THINK_END_CANDIDATES = ("</think>", "<｜thinking▁end｜>")


@dataclass(frozen=True)
class SpecialTokenIds:
    eos_token_ids: tuple[int, ...] = ()
    think_start_token_id: int = -1
    think_end_token_id: int = -2


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _detect_eos_from_config(config: dict) -> tuple[int, ...]:
    raw = config.get("eos_token_id")
    if raw is None:
        return ()
    if isinstance(raw, list):
        return tuple(int(x) for x in raw if isinstance(x, (int, float)))
    if isinstance(raw, (int, float)):
        return (int(raw),)
    return ()


def _scan_added_tokens_for_think(
    tokenizer_config: dict,
) -> tuple[int, int]:
    added = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(added, dict):
        return -1, -2
    start_id = -1
    end_id = -2
    for token_id_str, entry in added.items():
        if not isinstance(entry, dict):
            continue
        content = entry.get("content", "")
        if content == "<think>" and start_id < 0:
            try:
                start_id = int(token_id_str)
            except (ValueError, TypeError):
                pass
        elif content == "</think>" and end_id < 0:
            try:
                end_id = int(token_id_str)
            except (ValueError, TypeError):
                pass
        if start_id >= 0 and end_id >= 0:
            break
    return start_id, end_id


def detect_special_tokens(model_path: str | Path) -> SpecialTokenIds:
    """Detect special token IDs from model directory JSON files.

    Never raises — returns sentinel defaults on any failure.
    """
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        log.warning("model_path %s is not a directory", model_path)
        return SpecialTokenIds()

    eos_ids: tuple[int, ...] = ()
    think_start = -1
    think_end = -2

    config = _read_json(model_dir / "config.json")
    if config is not None:
        eos_ids = _detect_eos_from_config(config)
    else:
        log.warning("could not read config.json in %s", model_path)

    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    if tokenizer_config is not None:
        think_start, think_end = _scan_added_tokens_for_think(tokenizer_config)

    if think_start < 0 or think_end < 0:
        model_type = config.get("model_type", "") if config else ""
        known = _KNOWN_THINK_TOKENS.get(model_type)
        if known is not None:
            if think_start < 0:
                think_start = known[0]
            if think_end < 0:
                think_end = known[1]

    return SpecialTokenIds(
        eos_token_ids=eos_ids,
        think_start_token_id=think_start,
        think_end_token_id=think_end,
    )


def _search_vocab_for_think(tokenizer: object) -> tuple[int, int]:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return -1, -2
    unk_id = getattr(tokenizer, "unk_token_id", None)

    start_id = -1
    for candidate in _THINK_START_CANDIDATES:
        tid = convert(candidate)
        if tid != unk_id and isinstance(tid, int):
            start_id = tid
            break

    end_id = -2
    for candidate in _THINK_END_CANDIDATES:
        tid = convert(candidate)
        if tid != unk_id and isinstance(tid, int):
            end_id = tid
            break

    return start_id, end_id


class TokenizerWrapper:
    """HuggingFace tokenizer with special token detection."""

    def __init__(
        self,
        tokenizer_path: str = "auto",
        model_path: str | None = None,
    ) -> None:
        resolved = self._resolve_path(tokenizer_path, model_path)
        self._model_path = resolved

        from transformers import AutoTokenizer  # noqa: F811

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(resolved), trust_remote_code=True,
        )

        special = detect_special_tokens(resolved)

        think_start = special.think_start_token_id
        think_end = special.think_end_token_id
        if think_start < 0 or think_end < 0:
            vs, ve = _search_vocab_for_think(self._tokenizer)
            if think_start < 0:
                think_start = vs
            if think_end < 0:
                think_end = ve

        # EOS fallback: tokenizer directories without a config.json (or
        # with no eos_token_id in it) still carry an EOS on the HF
        # tokenizer object itself.  Serving needs SOME stop token or
        # generations only end at max_tokens (#71).
        eos_ids = special.eos_token_ids
        if not eos_ids:
            eid = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(eid, int):
                eos_ids = (eid,)

        self._special = SpecialTokenIds(
            eos_token_ids=eos_ids,
            think_start_token_id=think_start,
            think_end_token_id=think_end,
        )

    @staticmethod
    def _resolve_path(
        tokenizer_path: str, model_path: str | None,
    ) -> Path:
        if tokenizer_path == "auto":
            if model_path is None:
                raise ValueError(
                    "tokenizer_path is 'auto' but no model_path provided",
                )
            resolved = Path(model_path)
        else:
            resolved = Path(tokenizer_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"tokenizer path does not exist: {resolved}",
            )
        return resolved

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids))

    @property
    def special_tokens(self) -> SpecialTokenIds:
        return self._special

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._special.eos_token_ids

    @property
    def think_start_token_id(self) -> int:
        return self._special.think_start_token_id

    @property
    def think_end_token_id(self) -> int:
        return self._special.think_end_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def hf_tokenizer(self):
        """The underlying HuggingFace tokenizer (guided decoding builds
        its xgrammar TokenizerInfo from it — TD-SERVE-NAMED-TOOL-CHOICE)."""
        return self._tokenizer
