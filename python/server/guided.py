"""Guided (grammar-constrained) decoding for named tool_choice / "required".

Usage patterns in this module are ported (lean) from vLLM (ref/vllm;
Apache-2.0, Copyright contributors to the vLLM project — see
THIRD_PARTY_NOTICES.md).

TD-SERVE-NAMED-TOOL-CHOICE: compiles the request's tool JSON schemas into a
token-level grammar for the GLM-4.x/5.x tool-call wire format
(``<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``)
and drives per-step token masking so the model can only emit a valid call.

Grammar backend: **xgrammar structural tags** — xgrammar ships a builtin
``glm_4_7`` structural-tag model (``xgrammar.get_model_structural_tag``)
that expresses the GLM wrapper directly, including schema-typed
``<arg_value>`` content (``style="glm_xml"``) and the reasoning interplay
(``reasoning=True`` prefixes a free-form ``<think>``-section tag ending at
``</think>``, with the wire tags and stop tokens excluded inside it — so
think stays free-form, EOS is suppressed until the forced call closes, and
the grammar engages at the ``</think>`` boundary with no orchestrator-side
gating state).  No EBNF fallback is needed.  The builtin model name is
selected by the configured named tool parser (``structural_tag_model``):
"glm_4_7" (default) or "deepseek_v4" (the DSML wire format).

Usage patterns ported (lean) from ref/vllm:
  - ref/vllm/vllm/tool_parsers/structural_tag_registry.py
    (get_model_structural_tag protocol: dumped tool dicts + tool_choice
    normalization; the glm_4_7 builtin route)
  - ref/vllm/vllm/v1/structured_output/backend_xgrammar.py
    (TokenizerInfo.from_huggingface + GrammarCompiler + GrammarMatcher
    with rollback; bitmask fill/apply flow)

Layering: this module owns the xgrammar/torch/tokenizer dependencies; the
orchestrator only sees the duck-typed :class:`GuidedState` protocol
(``pick_and_accept`` / ``try_accept`` / ``rollback`` / ``completed``), so
python/orchestrator stays free of grammar imports (unit tests script a
fake state against the same protocol).
"""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np

__all__ = [
    "GuidedDecodingManager",
    "GuidedError",
    "GuidedGrammar",
    "GuidedState",
]

_GRAMMAR_CACHE_CAP = 32


class GuidedError(RuntimeError):
    """Guided decoding unavailable or grammar construction failed."""


class GuidedState:
    """Per-request grammar matcher + host-side masked sampling.

    Single-use: one state per generation.  All methods run on the
    orchestrator thread.
    """

    def __init__(self, matcher: Any, bitmask: Any, vocab_size: int) -> None:
        self._matcher = matcher
        self._bitmask = bitmask          # torch int32 [1, ceil(vocab/32)]
        self.vocab_size = int(vocab_size)
        self._rng: np.random.Generator | None = None

    # ── matcher protocol (duck-typed for the orchestrator) ──────────────

    @property
    def completed(self) -> bool:
        """True once the grammar can terminate (the forced call closed).
        The orchestrator stops emission here with finish "tool_calls"."""
        return bool(self._matcher.is_completed()
                    or self._matcher.is_terminated())

    def try_accept(self, token_id: int) -> bool:
        """Advance the matcher over one token; False = grammar violation
        (matcher state unchanged on rejection)."""
        return bool(self._matcher.accept_token(int(token_id)))

    def rollback(self, n: int) -> None:
        """Rewind the last ``n`` accepted tokens (STAGE B tentative
        draft-slot advances)."""
        if n > 0:
            self._matcher.rollback(int(n))

    # ── masked sampling ──────────────────────────────────────────────────

    def _allowed_bits(self) -> np.ndarray:
        """Boolean allow-vector [vocab] for the next position."""
        self._matcher.fill_next_token_bitmask(self._bitmask)
        words = self._bitmask.numpy()[0]
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")
        return bits[: self.vocab_size].astype(bool)

    def pick_and_accept(self, logits: np.ndarray, temperature: float = 0.0,
                        top_p: float = 1.0, top_k: int = 0,
                        seed: int = 42) -> int:
        """Mask ``logits`` with the grammar's next-token bitmask, pick a
        token (greedy argmax at temperature 0, else host-side
        temperature/top_k/top_p sampling), and advance the matcher.

        Because the pick comes from the mask, the accept cannot fail on a
        well-formed grammar; a failure raises (engine/grammar mismatch).
        """
        allowed = self._allowed_bits()
        if not allowed.any():
            raise GuidedError("guided: empty token mask (grammar dead-end)")
        masked = np.where(allowed, logits[: self.vocab_size], -np.inf)
        if temperature <= 0.0:
            tok = int(np.argmax(masked))
        else:
            tok = self._sample(masked, temperature, top_p, top_k, seed)
        if not self._matcher.accept_token(tok):
            raise GuidedError(
                f"guided: masked pick {tok} rejected by the matcher")
        return tok

    def _sample(self, masked: np.ndarray, temperature: float, top_p: float,
                top_k: int, seed: int) -> int:
        if self._rng is None:
            self._rng = np.random.default_rng(seed)
        logits = masked.astype(np.float64) / max(temperature, 1e-6)
        if top_k and top_k > 0:
            kth = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits >= kth, logits, -np.inf)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(-probs)
            csum = np.cumsum(probs[order])
            cut = int(np.searchsorted(csum, top_p) + 1)
            keep = order[:cut]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()
        return int(self._rng.choice(len(probs), p=probs))


class GuidedGrammar:
    """A compiled grammar (cacheable across requests with equal specs)."""

    def __init__(self, ctx: Any, xgr: Any, vocab_size: int) -> None:
        self._ctx = ctx
        self._xgr = xgr
        self._vocab_size = int(vocab_size)

    def new_state(self) -> GuidedState:
        # Rollback is unlimited in xgrammar >= 0.2 (the STAGE B tentative
        # draft-slot advances rely on it).
        matcher = self._xgr.GrammarMatcher(self._ctx)
        bitmask = self._xgr.allocate_token_bitmask(1, self._vocab_size)
        return GuidedState(matcher, bitmask, self._vocab_size)


class GuidedDecodingManager:
    """Lazy xgrammar compiler bound to the serving tokenizer.

    Built once at server construction from the HF tokenizer + engine vocab
    size; the (slow) TokenizerInfo/compiler construction and each grammar
    compile happen lazily on first guided request and are cached.
    """

    def __init__(self, hf_tokenizer: Any, vocab_size: int) -> None:
        self._hf_tokenizer = hf_tokenizer
        self._vocab_size = int(vocab_size)
        self._lock = threading.Lock()
        self._xgr: Any = None
        self._compiler: Any = None
        self._cache: dict[str, GuidedGrammar] = {}
        self._init_error: str | None = None

    def _ensure_compiler(self) -> None:
        if self._compiler is not None:
            return
        if self._init_error is not None:
            raise GuidedError(self._init_error)
        try:
            import xgrammar as xgr
            info = xgr.TokenizerInfo.from_huggingface(
                self._hf_tokenizer, vocab_size=self._vocab_size)
            self._compiler = xgr.GrammarCompiler(info)
            self._xgr = xgr
        except Exception as exc:  # noqa: BLE001 — surfaces as HTTP 400
            self._init_error = (
                f"guided decoding unavailable: {type(exc).__name__}: {exc}")
            raise GuidedError(self._init_error) from exc

    def build(self, tools: list[dict[str, Any]],
              tool_choice: dict[str, Any] | str,
              reasoning: bool,
              model: str = "glm_4_7") -> GuidedGrammar:
        """Compile (or fetch cached) the grammar for one request spec.

        ``tool_choice``: a named function object → that function only;
        ``"required"`` → union of all tools.  ``reasoning`` True prefixes
        the free-form ``<think>…</think>`` section (thinking-enabled
        templates pre-seed ``<think>`` in the prompt, so generation starts
        mid-reasoning; thinking=False templates pre-seed
        ``<think></think>`` and the grammar engages immediately).
        ``model``: the xgrammar builtin structural-tag model expressing the
        wire format — "glm_4_7" (default, the GLM wrapper) or
        "deepseek_v4" (the DSML wrapper; builtin confirmed in the
        installed wheel, builtin_structural_tag.py:1816) — selected by the
        configured named tool parser's ``structural_tag_model``.
        """
        key = json.dumps((model, tools, tool_choice, reasoning),
                         sort_keys=True)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            self._ensure_compiler()
            try:
                st = self._xgr.get_model_structural_tag(
                    model=model, tools=tools, tool_choice=tool_choice,
                    reasoning=reasoning)
                if st is None:
                    raise GuidedError(
                        "guided: structural tag builder returned nothing")
                ctx = self._compiler.compile_structural_tag(st)
            except GuidedError:
                raise
            except Exception as exc:  # noqa: BLE001 — bad schema etc. → 400
                raise GuidedError(
                    f"guided grammar construction failed: "
                    f"{type(exc).__name__}: {exc}") from exc
            grammar = GuidedGrammar(ctx, self._xgr, self._vocab_size)
            if len(self._cache) >= _GRAMMAR_CACHE_CAP:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = grammar
            return grammar
