"""DeepSeek-V4-Flash dspark speculation through the ring bridge (ticket J).

Boots the production engine on the shipped V4 config (target on GPU
position 0 = the 30-GiB 5090, dflash draft on position 1 = the second
5090) and runs the SAME greedy decode twice over the France golden prompt:

  * PLAIN    — teacher-forced prompt feed + B=1 greedy decode (the
               ticket-H golden harness shape; the 0.3-0.6 tok/s baseline).
  * DSPARK   — teacher-forced prompt feed + the dsp52 batched-verify round
               structure (draft step -> DSP-9 truncation -> (1+g)-row V4
               micro-chunk verify -> greedy longest-prefix accept + bonus).

GATES (the ticket-J acceptance set):
  * LOSSLESS (decisive): the speculative committed trajectory is
    TOKEN-IDENTICAL to the plain greedy trajectory (INV-DSPARK-LOSSLESS —
    this also exercises the executor rewind snapshots end-to-end).
  * acceptance > 0 over the run.
  * walls reported honestly for both arms (spec multiplies
    tokens/fetch-round on the expert-streaming wall).

Run (GPU window — GLM server STOPPED; 2 GPUs visible)::

  DSP4_SPEC=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
      .venv/bin/python -m pytest tests/integration/test_dsp4_bridge.py -s

Knobs: DSP4_TOKENS (default 24), DSP4_CONF_THRESH (default 0.1),
DSP4_GAMMA (default = config speculative_tokens).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import time

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CONFIG = _PROJECT_ROOT / "test-data" / "config" / "deepseek_v4_flash_gguf.json"

# Ticket-H France golden prompt: "The capital of France is" -> " Paris".
_PROMPT = [671, 6102, 294, 8760, 344]

pytestmark = pytest.mark.skipif(
    os.environ.get("DSP4_SPEC") != "1",
    reason="V4 dspark bridge run is opt-in (DSP4_SPEC=1; needs the V4 GGUF "
           "+ two free 5090s)")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v and v.lstrip("-").isdigit() else default


def _feed_prompt_v4(bridge, seq_id: int, prompt: list[int]) -> None:
    """Teacher-forced per-token prompt feed (TD-V4-CHUNK-PREFILL)."""
    t0 = time.monotonic()
    for i, tok in enumerate(prompt[:-1]):
        r = bridge.decode_step_fetch_and_run(tok, seq_id, i, None)
        assert 0 <= r.sampled_token < bridge.vocab_size
    print(f"  [dsp4] prompt fed: {len(prompt) - 1} tokens in "
          f"{time.monotonic() - t0:.1f} s", flush=True)


def _plain_v4(bridge, seq_id: int, prompt: list[int],
              n: int) -> tuple[list[int], float]:
    bridge.create_sequence(seq_id, len(prompt))
    try:
        _feed_prompt_v4(bridge, seq_id, prompt)
        out: list[int] = []
        token, pos = prompt[-1], len(prompt) - 1
        t0 = time.monotonic()
        for i in range(n):
            r = bridge.decode_step_fetch_and_run(token, seq_id, pos + i,
                                                 None)
            assert 0 <= r.sampled_token < bridge.vocab_size
            assert math.isfinite(r.top1_prob) and math.isfinite(r.entropy)
            out.append(r.sampled_token)
            token = r.sampled_token
            print(f"  [dsp4-plain] step {i:3d}: token={token:6d} "
                  f"ms={r.timings.total_ms:.0f}", flush=True)
        wall = time.monotonic() - t0
        return out, wall
    finally:
        bridge.free_sequence(seq_id)


def _spec_v4(bridge, seq_id: int, prompt: list[int], n: int, gamma: int,
             conf_thresh: float) -> tuple[list[int], float, dict]:
    """The dsp52 batched-verify round structure over the V4 micro-chunk
    verify arm (bridge.spec_decode.run_speculative_loop parity, minus the
    chunked prefill V4 cannot take)."""
    with_conf = conf_thresh > 0.0
    stats = {"rounds": 0, "proposed": 0, "accepted": 0,
             "fallback_rounds": 0}
    committed: list[int] = []
    bridge.create_sequence(seq_id, len(prompt))
    try:
        _feed_prompt_v4(bridge, seq_id, prompt)
        t0 = time.monotonic()
        # Seed feed: one plain step arms the draft context (aux export).
        r0 = bridge.decode_step_fetch_and_run(prompt[-1], seq_id,
                                              len(prompt) - 1, None)
        committed.append(r0.sampled_token)
        anchor, fed = r0.sampled_token, len(prompt)
        try:
            while len(committed) < n:
                draft, confs = bridge.dspark_draft_step(seq_id, anchor, fed,
                                                        gamma, with_conf)
                g_use = gamma
                if with_conf:
                    cum, g_use = 1.0, 0
                    while g_use < gamma:
                        cum *= confs[g_use]
                        if cum < conf_thresh:
                            break
                        g_use += 1
                j = 0
                if g_use == 0:
                    rr = bridge.decode_step_fetch_and_run(anchor, seq_id,
                                                          fed, None)
                    committed.append(rr.sampled_token)
                    fed += 1
                    bonus = rr.sampled_token
                    stats["fallback_rounds"] += 1
                else:
                    row_toks = [anchor] + draft[:g_use]
                    vr = bridge.verify_step_fetch_and_run(row_toks, seq_id,
                                                          fed, None)
                    while j < g_use and vr.argmax[j] == draft[j]:
                        j += 1
                    committed.extend(draft[:j])
                    bonus = vr.argmax[j]
                    committed.append(bonus)
                    fed += j + 1
                anchor = bonus
                stats["rounds"] += 1
                stats["proposed"] += g_use
                stats["accepted"] += j
                print(f"  [dsp4-spec] round {stats['rounds']:3d}: "
                      f"accepted={j}/{g_use} committed={len(committed)}",
                      flush=True)
        finally:
            bridge.drain_pending_dspark(gamma)
        wall = time.monotonic() - t0
        return committed, wall, stats
    finally:
        bridge.free_sequence(seq_id)


def test_dsp4_bridge_speculative_lossless_and_live():
    import layerstorm_engine
    from bridge.ring_bridge import EngineBridge, fastbridge_active

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        pytest.skip("V4 dspark bridge requires CUDA_VISIBLE_DEVICES=2,3 "
                    "(target 5090 + draft 5090)")
    assert _CONFIG.exists()
    cfg = json.loads(_CONFIG.read_text())
    assert cfg["speculation"]["method"] == "dspark"
    gamma = _env_int("DSP4_GAMMA",
                     int(cfg["speculation"]["dspark"]["speculative_tokens"]))
    conf_thresh = float(os.environ.get("DSP4_CONF_THRESH", "0.1"))
    n = _env_int("DSP4_TOKENS", 24)

    print(f"\n[dsp4] booting {os.path.basename(str(_CONFIG))} "
          f"(gamma={gamma}, conf={conf_thresh}, tokens={n}; "
          f"fastbridge={'ON' if fastbridge_active() else 'off'}) ...",
          flush=True)
    t0 = time.monotonic()
    info = layerstorm_engine.start_engine(str(_CONFIG))
    try:
        print(f"[dsp4] engine up in {time.monotonic() - t0:.0f}s: "
              f"layers={info.num_layers} gpus={info.num_gpus} "
              f"vocab={info.vocab_size}", flush=True)
        assert info.num_gpus == 2, "target + draft GPU expected"

        buffer_ids = layerstorm_engine.query_buffer_ids()
        hidden_buf = logits_buf = 0
        for name, bid in buffer_ids.items():
            if name.startswith("hidden_state.attn.rank0"):
                hidden_buf = int(bid)
            elif name.startswith("logits_scratch.pos0"):
                logits_buf = int(bid)
        assert hidden_buf and logits_buf

        bridge = EngineBridge(info, vocab_size=int(info.vocab_size),
                              first_moe_layer=0,
                              hidden_buf_id=hidden_buf,
                              logits_buf_id=logits_buf,
                              route_arm="act", use_far=False)
        bridge.decode_timeout_us = 200_000_000
        # Experts stay on the target GPU (position 0); position 1 is the
        # dspark draft host (an e%2 spread forks the golden trajectory).
        bridge.moe_gpus = 1

        plain_tokens, plain_wall = _plain_v4(bridge, 1, _PROMPT, n)
        print(f"[dsp4] PLAIN: {n} tokens in {plain_wall:.1f} s "
              f"({n / plain_wall:.3f} tok/s)", flush=True)
        # Ticket-H golden regression: "The capital of France is" -> " Paris.
        # The capital of Spain is" (llama.cpp-matched continuation).
        assert plain_tokens[:4] == [11111, 16, 455, 6102], (
            f"plain V4 greedy diverged from the ticket-H golden: "
            f"{plain_tokens[:8]}")

        spec_tokens, spec_wall, st = _spec_v4(bridge, 2, _PROMPT, n, gamma,
                                              conf_thresh)
        n_common = min(len(plain_tokens), len(spec_tokens))
        acc_rate = (st["accepted"] / st["proposed"]
                    if st["proposed"] else 0.0)
        tau = (len(spec_tokens) / st["rounds"] if st["rounds"] else 0.0)
        print(f"[dsp4] SPEC : {len(spec_tokens)} tokens in "
              f"{spec_wall:.1f} s ({len(spec_tokens) / spec_wall:.3f} "
              f"tok/s); rounds={st['rounds']} "
              f"fallback={st['fallback_rounds']} acc={acc_rate:.4f} "
              f"tau={tau:.3f}", flush=True)
        print(f"[dsp4] speedup vs plain: "
              f"{(len(spec_tokens) / spec_wall) / (n / plain_wall):.3f}x",
              flush=True)

        # GATE (decisive): LOSSLESS — token identity vs plain greedy.
        assert spec_tokens[:n_common] == plain_tokens[:n_common], (
            f"speculative trajectory diverged from plain greedy:\n"
            f"  plain={plain_tokens[:n_common]}\n"
            f"  spec ={spec_tokens[:n_common]}")
        # GATE: speculation LIVE.
        assert st["accepted"] > 0, "no draft token accepted over the run"
        assert st["rounds"] > 0
    finally:
        layerstorm_engine.stop_engine()
