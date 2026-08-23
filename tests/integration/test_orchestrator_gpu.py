"""GPU smoke of the bridge-based production Orchestrator on the champion
stack (ORCH_PY=1; 4-GPU GLM-5.2 + DSpark ckpt, warm arena ~3 min).

Boots the engine through Orchestrator.boot() with the dsp52 champion
config (reused from test_dsp52_bridge's builder — same feature deltas the
C++ champion runs), submits one prompt-fed greedy request, and asserts
the serving contract: streamed tokens in order, EOS/length finish, sane
speculation stats.  This is the pre-endpoint validation of the successor
core (the OpenAI endpoint consumes Orchestrator next).

RUN:  ORCH_PY=1 .venv/bin/python -m pytest \
          tests/integration/test_orchestrator_gpu.py -s
"""

from __future__ import annotations

import json
import os

import pytest

# Importing the bridge test applies its _setup_env() contract (REEF arm,
# CUDA ordering, IPC pin, arena attach) and brings the config builder.
from test_dsp52_bridge import (
    _PROMPT_DEFAULT,
    _build_dsp52_config,
    _dsp52_ckpt_dir,
    _ep4_gpus_ok,
    _load_token_file,
)

from orchestrator.orchestrator import InferenceRequest, Orchestrator

GEN_TOKENS = 40


@pytest.mark.skipif(os.environ.get("ORCH_PY") != "1",
                    reason="set ORCH_PY=1 to run (4-GPU GLM-5.2 smoke)")
@pytest.mark.skipif(not (_dsp52_ckpt_dir() / "config.json").exists(),
                    reason="DSpark checkpoint not present")
@pytest.mark.skipif(not _PROMPT_DEFAULT.exists(),
                    reason="champion prompt fixture not present")
@pytest.mark.skipif(not _ep4_gpus_ok(),
                    reason="EP=4 needs 4 idle GPUs (2x >=28 GiB + 2x >=14)")
def test_orchestrator_champion_smoke():
    cfg, gamma, _block = _build_dsp52_config()
    cfg_path = "/tmp/orch_gpu_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    prompt = _load_token_file(_PROMPT_DEFAULT)[:256]
    orch = Orchestrator.boot(cfg_path, conf_thresh=0.1)
    try:
        assert orch.spec.enabled and orch.spec.gamma == gamma
        assert orch.spec.conf_thresh == 0.1

        streamed: list[int] = []
        done: list[tuple] = []
        orch.submit_request(InferenceRequest(
            request_id=1, prompt_token_ids=prompt, max_tokens=GEN_TOKENS,
            on_token=lambda rid, tok, lp: streamed.append(tok),
            on_complete=lambda rid, toks, reason, lp:
                done.append((rid, list(toks), reason))))
        assert orch._serve_next() is True

        rid, tokens, reason = done[0]
        assert rid == 1 and reason == "length"
        assert len(tokens) == GEN_TOKENS
        assert streamed == tokens, "on_token stream != final tokens"
        assert all(0 <= t < orch.bridge.vocab_size for t in tokens)

        st = orch.last_stats
        assert st.tokens == GEN_TOKENS
        assert st.proposed > 0 and 0 < st.accepted <= st.proposed
        print(f"\n  [orch-gpu] {GEN_TOKENS} tokens, rounds={st.rounds} "
              f"acc={st.accepted}/{st.proposed} "
              f"prefill={st.prefill_ms:.0f} ms "
              f"decode={st.decode_wall_ms:.0f} ms "
              f"-> {st.tok_per_s:.3f} tok/s", flush=True)
        print(f"  [orch-gpu] tokens: {tokens}", flush=True)
    finally:
        orch.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass


@pytest.mark.skipif(os.environ.get("ORCH_PY") != "1",
                    reason="set ORCH_PY=1 to run (4-GPU GLM-5.2 smoke)")
@pytest.mark.skipif(not (_dsp52_ckpt_dir() / "config.json").exists(),
                    reason="DSpark checkpoint not present")
@pytest.mark.skipif(not _PROMPT_DEFAULT.exists(),
                    reason="champion prompt fixture not present")
@pytest.mark.skipif(not _ep4_gpus_ok(),
                    reason="EP=4 needs 4 idle GPUs (2x >=28 GiB + 2x >=14)")
def test_prefix_cache_token_identity_and_prefill_reuse():
    """serving.prefix_cache GPU gate (INV-PREFIX-CACHE-1): a cache-hit
    request must produce TOKEN-IDENTICAL output to the cache-off control,
    and its prefill must reuse the holder's KV (wall <<, hit tokens
    reported). One engine boot, three requests."""
    cfg, gamma, _block = _build_dsp52_config()
    assert (cfg.get("serving", {}).get("prefix_cache", {})
            .get("enabled", True)), "schema default must be ON"
    cfg_path = "/tmp/orch_prefix_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    corpus = _load_token_file(_PROMPT_DEFAULT)
    prompt_a = corpus[:256]
    prompt_b = corpus[:288]          # shares prompt_a[:255] as prefix
    gen = 24

    orch = Orchestrator.boot(cfg_path, conf_thresh=0.1)
    try:
        def run(prompt: list[int]) -> tuple[list[int], object]:
            done: list[tuple] = []
            orch.submit_request(InferenceRequest(
                request_id=len(done) + 100, prompt_token_ids=prompt,
                max_tokens=gen,
                on_complete=lambda rid, toks, reason, lp:
                    done.append((list(toks), reason))))
            assert orch._serve_next() is True
            tokens, reason = done[0]
            assert reason in ("length", "stop")
            return tokens, orch.last_stats

        # Request A: cold — registers the 255-token prefix holder.
        tokens_a, st_a = run(prompt_a)
        assert st_a.prefix_hit_tokens == 0
        assert len(orch.prefix_cache._entries) >= 1

        # Request B: shares A's prefix — must fork + prefill only delta.
        tokens_b_on, st_on = run(prompt_b)
        # Grid-aligned registry: A pre=255 -> holder at 192.
        assert st_on.prefix_hit_tokens == 192
        print(f"\n  [prefix-gpu] A prefill {st_a.prefill_ms:.0f} ms "
              f"(255 tokens) vs B-hit prefill {st_on.prefill_ms:.0f} ms "
              f"({len(prompt_b) - 1 - st_on.prefix_hit_tokens} delta "
              f"tokens)", flush=True)
        assert st_on.prefill_ms < st_a.prefill_ms * 0.5, (
            f"prefix hit did not shorten prefill: {st_on.prefill_ms:.0f}"
            f" vs {st_a.prefill_ms:.0f} ms")

        # Cache-OFF control on the SAME engine: full re-prefill of B.
        saved, orch.prefix_cache = orch.prefix_cache, None
        try:
            tokens_b_off, st_off = run(prompt_b)
        finally:
            orch.prefix_cache = saved
        assert st_off.prefix_hit_tokens == 0

        # THE GATE (INV-PREFIX-CACHE-1): reused KV == recomputed KV.
        assert tokens_b_on == tokens_b_off, (
            f"prefix cache changed tokens!\n on={tokens_b_on}\n"
            f"off={tokens_b_off}")
        print(f"  [prefix-gpu] TOKEN IDENTITY: {len(tokens_b_on)} tokens "
              f"equal (on-hit vs off-control); "
              f"A tokens sample={tokens_a[:6]}", flush=True)
    finally:
        orch.stop_engine()
        try:
            os.remove(cfg_path)
        except OSError:
            pass
