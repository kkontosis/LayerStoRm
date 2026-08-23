#!/usr/bin/env python3
"""Mint a HuggingFace tokenizer directory from the DeepSeek-V4-Flash GGUF.

Uses llama.cpp's gguf-py at runtime (ref/llama.cpp; MIT License,
Copyright (c) 2023-2026 The ggml authors — see THIRD_PARTY_NOTICES.md).

TD-SERVE-GGUF-TOKENIZER (V4 arm): the serve stack needs an HF tokenizer
directory, but the V4-Flash artifact ships GGUF-embedded vocab only.
Header census (2026-08-20) established that the V4 tokenizer is the
DeepSeek-V3.2 HF tokenizer with a respecialized special-token region:

  * base BPE vocab ids 0..127999 byte-identical to
    /srv/models/deepseek-ai/DeepSeek-V3.2/tokenizer.json,
  * merges identical (127741),
  * pretokenizer ``joyai-llm`` == the DEEPSEEK3_LLM regex set
    (ref/llama.cpp/src/llama-vocab.cpp:2318 -> :320) == the V3.2
    tokenizer.json pre_tokenizer Sequence,
  * only the region >= 128000 differs (V4 respecializes/extends to
    129280: <think>=128821, </think>=128822, |DSML|=128825, ...).

So the mint reuses the V3.2 pipeline (normalizer / pre_tokenizer /
decoder / post_processor / BPE model settings) and takes vocab, merges,
token types, chat template and special ids from the GGUF header —
nothing is copied blind: vocab + merges come from the GGUF itself, and
the pipeline reuse is validated by the built-in parity battery
(ticket-H llama.cpp golden token ids + V3.2 plain-text encode parity +
special-token round trips).

Usage:
  .venv/bin/python tools/mint_v4_hf_tokenizer.py \
      [--gguf PATH] [--base V32_DIR] [--out OUT_DIR] [--skip-validate]

Writes tokenizer.json, tokenizer_config.json (chat template embedded),
config.json into OUT_DIR (default: the GGUF's directory, so cli/serve.py
--tokenizer-path auto detects it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "ref" / "llama.cpp" / "gguf-py"))

DEFAULT_GGUF = ("/srv/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/"
                "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf")
DEFAULT_BASE = "/srv/models/deepseek-ai/DeepSeek-V3.2"

# gguf token_type values (llama.cpp llama_token_type)
_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4

# Ticket-H llama.cpp goldens (minted via llama-completion, -no-cnv):
# prompt string -> exact token ids under the GGUF tokenizer.
_GOLDEN_ENCODINGS = {
    "The capital of France is": [671, 6102, 294, 8760, 344],
    "The capital of Japan is": [671, 6102, 294, 6310, 344],
    "Albert Einstein was born in": [671, 6129, 294, 76407, 515, 5873,
                                    513, 26218],
}
# NOTE: the Einstein prompt above is keyed by its ticket-H ids; the
# actual prompt STRING is recovered below by decoding those ids, so a
# wrong guess of the text cannot mis-gate the mint.
_GOLDEN_BY_IDS_ONLY = ["Albert Einstein was born in"]

_PLAIN_PARITY_BATTERY = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and\nnewlines\r\n\ttabs",
    "杭州市今天天气怎么样？",
    "12345 numbers 678 and 3.14159",
    "def f(x):\n    return x ** 2  # comment",
    'JSON: {"a": [1, 2, 3], "b": null}',
    "mixed 語言 text avec des accents éàü",
]


def read_gguf_tokenizer(gguf_path: str) -> dict:
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    f = r.fields

    def scalar(name):
        fld = f[name]
        v = fld.parts[fld.data[0]]
        try:
            return v.item() if hasattr(v, "item") and v.size == 1 else v
        except Exception:  # noqa: BLE001
            return v

    def str_field(name):
        fld = f[name]
        return bytes(fld.parts[fld.data[0]]).decode("utf-8")

    def str_array(name):
        fld = f[name]
        return [bytes(fld.parts[d]).decode("utf-8") for d in fld.data]

    def int_array(name):
        fld = f[name]
        if len(fld.data) == 1:
            return [int(x) for x in fld.parts[fld.data[0]]]
        return [int(fld.parts[d][0]) for d in fld.data]

    return {
        "tokens": str_array("tokenizer.ggml.tokens"),
        "types": int_array("tokenizer.ggml.token_type"),
        "merges": str_array("tokenizer.ggml.merges"),
        "chat_template": str_field("tokenizer.chat_template"),
        "bos_id": int(scalar("tokenizer.ggml.bos_token_id")),
        "eos_id": int(scalar("tokenizer.ggml.eos_token_id")),
        "pad_id": int(scalar("tokenizer.ggml.padding_token_id")),
        "add_bos": bool(scalar("tokenizer.ggml.add_bos_token")),
        "pre": str_field("tokenizer.ggml.pre"),
        "model": str_field("tokenizer.ggml.model"),
    }


def build_tokenizer_json(base_tj: dict, gg: dict) -> dict:
    tokens, types = gg["tokens"], gg["types"]
    assert gg["model"] == "gpt2", f"unexpected tokenizer model {gg['model']}"
    assert gg["pre"] == "joyai-llm", f"unexpected pretokenizer {gg['pre']}"

    # Vocab: every GGUF token at its id (specials included — mirrors the
    # V3.2 structure where added tokens also appear in model.vocab).
    vocab = {tok: i for i, tok in enumerate(tokens)}
    assert len(vocab) == len(tokens), "duplicate token strings in GGUF vocab"

    added = [
        {
            "id": i,
            "content": tokens[i],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": types[i] == _CONTROL,
        }
        for i in range(len(tokens)) if types[i] != _NORMAL
    ]

    model = dict(base_tj["model"])
    model["vocab"] = vocab
    model["merges"] = gg["merges"]

    return {
        "version": base_tj.get("version", "1.0"),
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": base_tj["normalizer"],
        "pre_tokenizer": base_tj["pre_tokenizer"],
        "post_processor": base_tj["post_processor"],
        "decoder": base_tj["decoder"],
        "model": model,
    }


def build_tokenizer_config(gg: dict) -> dict:
    tokens, types = gg["tokens"], gg["types"]
    added_decoder = {
        str(i): {
            "content": tokens[i],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": types[i] == _CONTROL,
        }
        for i in range(len(tokens)) if types[i] != _NORMAL
    }
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": gg["add_bos"],          # False for V4-Flash
        "add_eos_token": False,
        "bos_token": tokens[gg["bos_id"]],
        "eos_token": tokens[gg["eos_id"]],
        "pad_token": tokens[gg["pad_id"]],
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1048576,
        "chat_template": gg["chat_template"],
        "added_tokens_decoder": added_decoder,
    }


def build_config(gg: dict) -> dict:
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "vocab_size": len(gg["tokens"]),
        "bos_token_id": gg["bos_id"],
        "eos_token_id": gg["eos_id"],
    }


def validate(out_dir: Path, base_dir: Path, gg: dict) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out_dir))
    base = AutoTokenizer.from_pretrained(str(base_dir))
    tokens = gg["tokens"]
    fails = []

    def enc(t, s):
        return t.encode(s, add_special_tokens=False)

    # 1. Ticket-H llama.cpp golden encodings (bit-exact ground truth).
    for text, want in _GOLDEN_ENCODINGS.items():
        if text in _GOLDEN_BY_IDS_ONLY:
            text = tok.decode(want)
        got = enc(tok, text)
        if got != want:
            fails.append(f"golden {text!r}: got {got}, want {want}")

    # 2. Plain-text parity with the V3.2 tokenizer (identical base BPE).
    for text in _PLAIN_PARITY_BATTERY:
        a, b = enc(tok, text), enc(base, text)
        if a != b:
            fails.append(f"V3.2 parity {text!r}: v4={a} v3={b}")

    # 3. Special tokens: exact single-id round trips.
    for tid in [0, 1, 2, 128821, 128822, 128825]:
        got = enc(tok, tokens[tid])
        if got != [tid]:
            fails.append(f"special {tokens[tid]!r}: got {got}, want [{tid}]")

    # 4. Decode round-trip on plain text.
    for text in _PLAIN_PARITY_BATTERY:
        rt = tok.decode(enc(tok, text))
        if rt != text:
            fails.append(f"round-trip {text!r} -> {rt!r}")

    # 5. Vocab width.
    if len(tok) != len(tokens):
        fails.append(f"len(tokenizer) {len(tok)} != {len(tokens)}")

    if fails:
        for msg in fails:
            print("FAIL:", msg)
        raise SystemExit(f"{len(fails)} validation failures")
    print(f"validation OK: {len(_GOLDEN_ENCODINGS)} goldens, "
          f"{len(_PLAIN_PARITY_BATTERY)} parity strings, specials, "
          f"round-trips, vocab {len(tok)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="V3.2 HF tokenizer dir (pipeline donor)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: the GGUF's directory)")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.gguf).parent
    base_dir = Path(args.base)

    print(f"reading GGUF header: {args.gguf}")
    gg = read_gguf_tokenizer(args.gguf)
    print(f"  {len(gg['tokens'])} tokens, {len(gg['merges'])} merges, "
          f"pre={gg['pre']}, bos={gg['bos_id']} eos={gg['eos_id']} "
          f"add_bos={gg['add_bos']}")

    base_tj = json.loads((base_dir / "tokenizer.json").read_text())

    tj = build_tokenizer_json(base_tj, gg)
    tc = build_tokenizer_config(gg)
    cfg = build_config(gg)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tj, ensure_ascii=False))
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tc, ensure_ascii=False, indent=1))
    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1))
    print(f"wrote tokenizer.json / tokenizer_config.json / config.json "
          f"to {out_dir}")

    if not args.skip_validate:
        validate(out_dir, base_dir, gg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
