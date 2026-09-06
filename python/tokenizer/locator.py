"""Tokenizer-directory location: ONE precedence, shared by every resolver.

P-34 step 1 (spec/tickets/P-34_TOKENIZER_PATH_DECOUPLING.md): the engine
must never *require* the repo's ``test-data/`` tree to boot a model.  The
precedence, everywhere a tokenizer directory is resolved:

1. an explicit path (CLI flag / ``serving.tokenizer_path``) always wins;
2. the WEIGHTS directory (or a single-file checkpoint's parent);
3. a sibling of the weights dir whose name is a prefix of the weights dir
   name (``.../GLM-5.2-GGUF-Q4_K_XL`` -> ``.../GLM-5.2`` — still "beside
   the weights", longest matching name wins);
4. repo ``test-data/`` by sanitised model display name — FALLBACK ONLY.
   Callers MUST surface tier 4 loudly (a warning naming the resolved path):
   a repo test dir silently satisfying a production boot is a
   works-on-this-box-only failure.  ``test-data/`` stays valid for the
   repo's own tests; it is demoted, not removed.

Rationale: GGUF checkpoints carry no extractable HF tokenizer
(TD-SERVE-GGUF-TOKENIZER), so the tokenizer must live SOMEWHERE — the
supported place is next to the weights.
"""

from __future__ import annotations

import glob as _glob
import os
import re

TOKENIZER_MARKERS = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model")

# Resolution tiers, in precedence order (tier "explicit" is handled by the
# callers — an explicit path is used verbatim, never searched).
TIER_WEIGHTS = "weights"        # in the weights directory itself
TIER_SIBLING = "sibling"        # beside the weights (name-prefix sibling)
TIER_TEST_DATA = "test-data"    # repo test-data/ — fallback, log loudly


def sanitise_model_name(name: str) -> str:
    """Display name -> directory-comparable form (mirrors autoconfig)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return s.lower() or "model"


def _has_tokenizer(d: str) -> bool:
    return any(os.path.isfile(os.path.join(d, m)) for m in TOKENIZER_MARKERS)


def _search_chain(weights_path: str, has, model_name: str = "",
                  repo_root: str = "") -> tuple[str, str]:
    """The ONE search: weights dir -> name-prefix sibling -> repo
    test-data.  ``has(dir)`` decides whether a directory satisfies the
    caller.  Returns ``(directory, tier)`` or ``("", "")``."""
    w = os.path.abspath(str(weights_path))
    wdir = w if os.path.isdir(w) else os.path.dirname(w)

    # 2. the weights directory itself
    if has(wdir):
        return wdir, TIER_WEIGHTS

    # 3. name-prefix siblings, longest name first
    parent, base = os.path.dirname(wdir), os.path.basename(wdir)
    best = ""
    for sib in sorted(_glob.glob(os.path.join(parent, "*"))):
        if not os.path.isdir(sib) or os.path.abspath(sib) == wdir:
            continue
        name = os.path.basename(sib)
        if not base.startswith(name):
            continue
        if has(sib) and len(name) > len(os.path.basename(best)):
            best = sib
    if best:
        return best, TIER_SIBLING

    # 4. repo test-data/, matched by sanitised display name — FALLBACK
    if repo_root and model_name:
        want = sanitise_model_name(model_name)
        for cand in sorted(_glob.glob(os.path.join(
                os.path.abspath(repo_root), "test-data", "*"))):
            if not os.path.isdir(cand):
                continue
            if sanitise_model_name(os.path.basename(cand)) != want:
                continue
            if has(cand):
                return cand, TIER_TEST_DATA
    return "", ""


def locate_tokenizer_dir(weights_path: str, model_name: str = "",
                         repo_root: str = "") -> tuple[str, str]:
    """Search for HF tokenizer files per the P-34 precedence.

    Returns ``(directory, tier)`` — tier is one of TIER_WEIGHTS,
    TIER_SIBLING, TIER_TEST_DATA — or ``("", "")`` when nothing is found.
    Explicit paths are the callers' business and never reach this function.
    """
    return _search_chain(weights_path, _has_tokenizer,
                         model_name=model_name, repo_root=repo_root)


def locate_model_asset(weights_path: str, filename: str,
                       model_name: str = "",
                       repo_root: str = "") -> tuple[str, str]:
    """Locate ONE model-metadata file (chat_template.jinja, config.json,
    generation_config.json, ...) with the same precedence as the
    tokenizer.  Returns ``(file path, tier)`` or ``("", "")``.  Tier
    TIER_TEST_DATA must be surfaced loudly by the caller."""
    d, tier = _search_chain(
        weights_path, lambda x: os.path.isfile(os.path.join(x, filename)),
        model_name=model_name, repo_root=repo_root)
    return (os.path.join(d, filename), tier) if d else ("", "")
