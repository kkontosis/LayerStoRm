"""Plain + DSpark-speculative decode loops over EngineBridge.

Faithful ports of ``run_plain_loop`` / ``run_speculative_loop`` from
tests/integration/dsp52_test.cpp (the token%-champion speculative arm and
its A/B plain baseline), with identical round structure and accounting:

Speculative round (batched verify, ``DSP52_VB=batched``):
  (a) DRAFT   — one D_CMD_RUN_DSPARK_STEP (γ block, DSP-5/DSP-6 readback);
  (b) TRUNCATE— DSP-9 static-threshold: keep the prefix with
                cumprod(c_1..c_k) >= conf_thresh (0 slots → hybrid plain
                fallback step);
  (c) VERIFY  — one (1+g_use)-row chunk [anchor@fed, d_0@fed+1, ...];
  (d) ACCEPT  — greedy longest matching prefix + bonus; fed += j+1.

Overlap round (``DSP52_OVERLAP=1``, the champion shape): the draft runs
async UNDER an always-committing plain decode step; when draft[0] matches
the plain result the REMAINING slots verify in one (g_use)-row chunk.

Sequential verify (``vb="seq"``): early-stop teacher-forced feeds through
the same plain decode step — the lossless-by-construction default arm.

Teacher-forcing (`forced` — DSP52_FORCE_TRAJ semantics): the committed
stream is replaced by the pre-recorded trajectory (walls honest, every
step/draft/chunk still executes); acceptance measures the drafts against
the forced stream.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

from bridge.ring_bridge import BridgeError, EngineBridge, GpuLru


@dataclass
class PlainAgg:
    total_ms: float = 0.0
    embed: float = 0.0
    attn: float = 0.0
    moe: float = 0.0
    head: float = 0.0
    sample: float = 0.0
    lookups: int = 0
    nan_count: int = 0
    per_token_ms: list[float] = field(default_factory=list)


@dataclass
class SpecStats:
    rounds: int = 0
    fallback_rounds: int = 0        # conf-truncated-to-0 rounds (plain step)
    trunc_slots: int = 0            # draft slots dropped by conf_thresh
    proposed: int = 0
    accepted: int = 0
    wall_ms: float = 0.0            # whole loop incl. seed (excl. prefill)
    draft_ms: float = 0.0           # overlap: only the EXPOSED wait
    overlap_plain_ms: float = 0.0
    verify_ms: float = 0.0
    attn_ms: float = 0.0
    moe_ms: float = 0.0
    lookups: int = 0
    nan_count: int = 0


def _feed_prompt(bridge: EngineBridge, seq_id: int, prompt: list[int],
                 lrus: list[GpuLru] | None) -> None:
    """Chunked prefill of the first L-1 prompt tokens (64-token chunks);
    the LAST prompt token is fed by the caller as the seed decode step."""
    chunk = 64
    pre = len(prompt) - 1
    t0 = time.monotonic()
    pos = 0
    while pos < pre:
        n = min(chunk, pre - pos)
        bridge.prefill_chunk_fetch_and_run(prompt[pos:pos + n], seq_id, pos,
                                           lrus)
        pos += n
    print(f"  [bridge] prompt prefill: {pre} tokens in "
          f"{time.monotonic() - t0:.1f} s (chunk {chunk}; excluded from "
          f"perf stats)", flush=True)


def run_plain_loop(bridge: EngineBridge, seq_id: int, num_tokens: int,
                   seed_token: int, lrus: list[GpuLru] | None,
                   agg: PlainAgg, *, prompt: list[int] | None = None,
                   forced: list[int] | None = None,
                   dump_all: bool = False,
                   h2d_base_cb=None) -> list[int]:
    """The plain keeper loop (DSP52_SPEC=0 arm). Returns the committed
    trajectory (num_tokens generated tokens)."""
    out: list[int] = []
    prompt_len = len(prompt) if prompt else 0
    bridge.create_sequence(seq_id, prompt_len if prompt_len else 1)
    try:
        pos0, token = 0, seed_token
        if prompt_len:
            _feed_prompt(bridge, seq_id, prompt, lrus)
            token = prompt[-1]
            pos0 = prompt_len - 1
        if h2d_base_cb is not None:
            h2d_base_cb()  # snapshot the miss baseline AFTER any prefill
        force_match = 0
        for i in range(num_tokens):
            r = bridge.decode_step_fetch_and_run(token, seq_id, pos0 + i,
                                                 lrus)
            if r.sampled_token >= bridge.vocab_size:
                raise BridgeError(f"invalid token at step {i}: "
                                  f"{r.sampled_token}")
            if not (math.isfinite(r.top1_prob)
                    and math.isfinite(r.entropy)):
                agg.nan_count += 1
            agg.total_ms += r.timings.total_ms
            agg.per_token_ms.append(r.timings.total_ms)
            agg.embed += r.timings.embedding_ms
            agg.head += r.timings.output_head_ms
            agg.sample += r.timings.sample_ms
            agg.attn += sum(r.timings.attention_ms)
            agg.moe += sum(r.timings.moe_ms)
            agg.lookups += r.moe_lookups
            if dump_all or i == 0 or (i + 1) % 10 == 0:
                print(f"  step {i:3d}: token={r.sampled_token:6d}  "
                      f"prob={r.top1_prob:.3f}  "
                      f"ms={r.timings.total_ms:.3f}", flush=True)
            nxt = r.sampled_token
            if forced and i < len(forced):
                if r.sampled_token == forced[i]:
                    force_match += 1
                nxt = forced[i]
            out.append(nxt)
            token = nxt
        if forced:
            print(f"  [bridge] FORCE_TRAJ plain: model matched "
                  f"{force_match}/{num_tokens} forced tokens", flush=True)
    finally:
        bridge.free_sequence(seq_id)
    return out


def run_speculative_loop(bridge: EngineBridge, seq_id: int, gen_tokens: int,
                         seed_token: int, lrus: list[GpuLru] | None,
                         st: SpecStats, *, gamma: int,
                         conf_thresh: float = 0.0,
                         vb: str = "batched",
                         overlap: bool = False,
                         prefetch: bool = False,
                         prompt: list[int] | None = None,
                         forced: list[int] | None = None,
                         h2d_base_cb=None) -> list[int]:
    """The dsp52 speculative decode loop. Commits AT LEAST `gen_tokens`
    tokens (the last round's bonus may overshoot); returns the committed
    trajectory (first element = first generated token)."""
    committed: list[int] = []
    R = gamma + 1
    if R > bridge.moe_batch_capacity:
        raise BridgeError(
            f"verify rows R={R} exceed engine MoE batch capacity "
            f"{bridge.moe_batch_capacity}")
    if overlap and vb != "batched":
        print("  [bridge] WARNING: overlap requires vb=batched — ignored",
              flush=True)
        overlap = False
    with_conf = conf_thresh > 0.0

    prompt_len = len(prompt) if prompt else 0
    bridge.create_sequence(seq_id, prompt_len if prompt_len else 1)
    try:
        seed, seed_pos = seed_token, 0
        if prompt_len:
            _feed_prompt(bridge, seq_id, prompt, lrus)
            seed = prompt[-1]
            seed_pos = prompt_len - 1
        if h2d_base_cb is not None:
            h2d_base_cb()
        loop_start = time.monotonic()

        # Seed feed: one plain decode step → first generated token; the
        # aux export captures the row, arming the draft context.
        r0 = bridge.decode_step_fetch_and_run(seed, seq_id, seed_pos, lrus)
        st.lookups += r0.moe_lookups
        if not (math.isfinite(r0.top1_prob) and math.isfinite(r0.entropy)):
            st.nan_count += 1

        def ftok(idx: int, model_tok: int) -> int:
            return forced[idx] if (forced and idx < len(forced)) \
                else model_tok

        committed.append(ftok(0, r0.sampled_token))
        print(f"  seed step: token={r0.sampled_token}  "
              f"prob={r0.top1_prob:.3f}  ms={r0.timings.total_ms:.3f}",
              flush=True)

        # ── LS_BRIDGE_ROUND_CSV (bridge-gap ledger, TD-BRIDGE-CPP-GAP): one
        # row per seed step / overlap macro-round with the per-section walls
        # — OFF unless the env var names a file (zero champion overhead:
        # timings written already exist). Columns mirror the C++ test's
        # DSP52_ROUND_CSV (dsp52_test.cpp) for the paired ledger.
        round_csv = None
        rc = os.environ.get("LS_BRIDGE_ROUND_CSV")
        if rc:
            round_csv = open(rc, "w")
            round_csv.write(
                "kind,round,wall_ms,plain_total,plain_embed,plain_layers,"
                "plain_head,plain_sample,draft_exposed,verify_total,"
                "verify_embed,verify_layers,verify_head,gap_ms,j,g_use\n")
            t0 = r0.timings
            round_csv.write(
                f"seed,0,{t0.total_ms:.6f},{t0.total_ms:.6f},"
                f"{t0.embedding_ms:.6f},"
                f"{sum(t0.attention_ms) + sum(t0.moe_ms):.6f},"
                f"{t0.output_head_ms:.6f},{t0.sample_ms:.6f},"
                f"0,0,0,0,0,0,0,0\n")

        anchor = committed[-1]
        fed = seed_pos + 1

        def absorb_chunk(vr) -> None:
            st.verify_ms += vr.timings.total_ms
            st.lookups += vr.moe_lookups
            st.attn_ms += sum(vr.timings.attention_ms)
            st.moe_ms += sum(vr.timings.moe_ms)
            if not (math.isfinite(vr.top1_prob)
                    and math.isfinite(vr.entropy)):
                st.nan_count += 1

        def conf_trace(j_acc: int, g_use: int, confs: list[float]) -> None:
            if conf_thresh <= 0.0:
                return
            cs = " ".join(f"{c:.6f}" for c in confs)
            print(f"  [bridge-conf] round {st.rounds}: c_k=[ {cs} ] "
                  f"g_use={g_use} accepted={j_acc}", flush=True)

        try:
            while len(committed) < gen_tokens:
                if overlap:
                    # ── overlap macro-round (lever 2, champion shape) ────
                    t_round0 = time.monotonic()
                    bridge.dspark_send_async(seq_id, anchor, fed, gamma)
                    rr = bridge.decode_step_fetch_and_run(anchor, seq_id,
                                                          fed, lrus)
                    st.overlap_plain_ms += rr.timings.total_ms
                    st.lookups += rr.moe_lookups
                    st.attn_ms += sum(rr.timings.attention_ms)
                    st.moe_ms += sum(rr.timings.moe_ms)
                    if not (math.isfinite(rr.top1_prob)
                            and math.isfinite(rr.entropy)):
                        st.nan_count += 1
                    t = ftok(len(committed), rr.sampled_token)
                    committed.append(t)
                    fed += 1
                    tw = time.monotonic()
                    draft, confs = bridge.dspark_collect_async(gamma,
                                                               with_conf)
                    draft_exposed = (time.monotonic() - tw) * 1e3
                    st.draft_ms += draft_exposed
                    # Truncation over the REMAINING slots: slot 0 is free
                    # against the plain result; slot k>=1 survives at
                    # prod(c_1..c_k) conditional on slot 0 accepted.
                    g_use = gamma
                    if with_conf:
                        cum, g_use = 1.0, 1
                        while g_use < gamma:
                            cum *= confs[g_use]
                            if cum < conf_thresh:
                                break
                            g_use += 1
                        st.trunc_slots += gamma - g_use
                    j = 0
                    round_verify_ms = 0.0
                    v_embed = v_layers = v_head = 0.0
                    if t == draft[0] and g_use >= 2:
                        row_toks = [t] + draft[1:g_use]
                        vr = bridge.verify_step_fetch_and_run(
                            row_toks, seq_id, fed, lrus, prefetch=prefetch)
                        absorb_chunk(vr)
                        round_verify_ms = vr.timings.total_ms
                        v_embed = vr.timings.embedding_ms
                        v_layers = (sum(vr.timings.attention_ms)
                                    + sum(vr.timings.moe_ms))
                        v_head = vr.timings.output_head_ms
                        j2 = 0
                        if forced:
                            base = len(committed)
                            while (j2 + 1 < g_use
                                   and draft[j2 + 1] == ftok(
                                       base + j2, vr.argmax[j2])):
                                j2 += 1
                        else:
                            while (j2 + 1 < g_use
                                   and vr.argmax[j2] == draft[j2 + 1]):
                                j2 += 1
                        committed.extend(draft[1:j2 + 1])
                        bonus = ftok(len(committed), vr.argmax[j2])
                        committed.append(bonus)
                        fed += j2 + 1
                        anchor = bonus
                        j = 1 + j2
                    else:
                        if t == draft[0]:
                            j = 1  # matched but nothing left to verify
                        anchor = t
                    st.rounds += 1
                    st.proposed += g_use
                    st.accepted += j
                    print(f"  round {st.rounds:3d} (overlap): accepted="
                          f"{j}/{g_use}  committed={len(committed)}  "
                          f"plain ms={rr.timings.total_ms:.3f}  "
                          f"verify ms={round_verify_ms:.3f}", flush=True)
                    conf_trace(j, g_use, confs)
                    if round_csv is not None:
                        wall = (time.monotonic() - t_round0) * 1e3
                        tp = rr.timings
                        round_csv.write(
                            f"overlap,{st.rounds},{wall:.6f},"
                            f"{tp.total_ms:.6f},{tp.embedding_ms:.6f},"
                            f"{sum(tp.attention_ms) + sum(tp.moe_ms):.6f},"
                            f"{tp.output_head_ms:.6f},{tp.sample_ms:.6f},"
                            f"{draft_exposed:.6f},{round_verify_ms:.6f},"
                            f"{v_embed:.6f},{v_layers:.6f},{v_head:.6f},"
                            f"{wall - tp.total_ms - draft_exposed - round_verify_ms:.6f},"
                            f"{j},{g_use}\n")
                    continue

                # (a) DRAFT — one whole-γ DSpark block.
                td = time.monotonic()
                draft, confs = bridge.dspark_draft_step(seq_id, anchor, fed,
                                                        gamma, with_conf)
                st.draft_ms += (time.monotonic() - td) * 1e3

                # (b) DSP-9 static-threshold truncation.
                g_use = gamma
                if with_conf:
                    cum, g_use = 1.0, 0
                    while g_use < gamma:
                        cum *= confs[g_use]
                        if cum < conf_thresh:
                            break
                        g_use += 1
                    st.trunc_slots += gamma - g_use

                j = 0
                round_verify_ms = 0.0
                if g_use == 0:
                    # Hybrid AR fallback: one plain step, always commits.
                    rr = bridge.decode_step_fetch_and_run(anchor, seq_id,
                                                          fed, lrus)
                    st.verify_ms += rr.timings.total_ms
                    round_verify_ms = rr.timings.total_ms
                    st.lookups += rr.moe_lookups
                    st.attn_ms += sum(rr.timings.attention_ms)
                    st.moe_ms += sum(rr.timings.moe_ms)
                    if not (math.isfinite(rr.top1_prob)
                            and math.isfinite(rr.entropy)):
                        st.nan_count += 1
                    bonus = ftok(len(committed), rr.sampled_token)
                    committed.append(bonus)
                    fed += 1
                    st.fallback_rounds += 1
                elif vb == "batched":
                    # (c) batched verify — one (1+g_use)-row chunk.
                    row_toks = [anchor] + draft[:g_use]
                    vr = bridge.verify_step_fetch_and_run(
                        row_toks, seq_id, fed, lrus, prefetch=prefetch)
                    absorb_chunk(vr)
                    round_verify_ms = vr.timings.total_ms
                    # (d) longest matching prefix + bonus.
                    if forced:
                        base = len(committed)
                        while (j < g_use and draft[j] == ftok(
                                base + j, vr.argmax[j])):
                            j += 1
                    else:
                        while j < g_use and vr.argmax[j] == draft[j]:
                            j += 1
                    committed.extend(draft[:j])
                    bonus = ftok(len(committed), vr.argmax[j])
                    committed.append(bonus)
                    fed += j + 1
                else:
                    # Sequential early-stop verify (lossless default arm).
                    feed = anchor
                    targets: list[int] = []
                    seq_base = len(committed)
                    for k in range(g_use + 1):
                        rr = bridge.decode_step_fetch_and_run(
                            feed, seq_id, fed + k, lrus)
                        st.verify_ms += rr.timings.total_ms
                        round_verify_ms += rr.timings.total_ms
                        st.lookups += rr.moe_lookups
                        st.attn_ms += sum(rr.timings.attention_ms)
                        st.moe_ms += sum(rr.timings.moe_ms)
                        if not (math.isfinite(rr.top1_prob)
                                and math.isfinite(rr.entropy)):
                            st.nan_count += 1
                        tgt = ftok(seq_base + len(targets),
                                   rr.sampled_token)
                        targets.append(tgt)
                        if k == g_use:
                            break  # bonus after full acceptance
                        if tgt != draft[k]:
                            break  # mismatch → its output = bonus
                        feed = tgt
                    j = len(targets) - 1
                    committed.extend(targets)
                    bonus = targets[-1]
                    fed += j + 1

                anchor = bonus
                st.rounds += 1
                st.proposed += g_use
                st.accepted += j
                print(f"  round {st.rounds:3d}: accepted={j}/{g_use}  "
                      f"bonus={bonus}  committed={len(committed)}  "
                      f"verify ms={round_verify_ms:.3f}", flush=True)
                conf_trace(j, g_use, confs)
        finally:
            bridge.drain_pending_dspark(gamma)
            if round_csv is not None:
                round_csv.close()

        st.wall_ms = (time.monotonic() - loop_start) * 1e3
    finally:
        bridge.free_sequence(seq_id)
    return committed
