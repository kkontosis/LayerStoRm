"""Bridge ring/loop unit tests against a scripted Python daemon (no CUDA).

A fake daemon thread consumes the command ring and produces completions
over a Python-owned memory block laid out like the engine's IPC region,
emulating a deterministic Markov target model:

    f(t) = (t * 1103515245 + 12345) % VOCAB

Greedy decode from seed s is then the chain f(s), f²(s), ...; the DSpark
draft proposes that chain with a per-round error injected at a cycling
slot.  The load-bearing gate is the LOSSLESS INVARIANT: whatever the
draft quality, every speculative arm (batched / seq / overlap / conf-
truncated) must commit exactly the greedy chain — the same invariant
INV-DSPARK-LOSSLESS pins on the real engine.

Also exercised: ring wrap-around, fire-and-forget completion dropping,
the async draft stash, sideband routing-export dedup, the 13c-2.0 LRU
victim map, and Cython (_fastbridge) vs pure-ctypes parity (both paths
run when the fast module is built).
"""

from __future__ import annotations

import ctypes
import threading

import pytest

from bridge import ring_bridge
from bridge.protocol import (
    CMD_EMBEDDING_LOOKUP,
    CMD_OUTPUT_HEAD,
    CMD_SAMPLE_TOKENS,
    CMD_SEQ_CREATE,
    CMD_SEQ_FORK,
    CMD_SEQ_FREE,
    CMP_COMPUTE_DONE,
    CMP_ERROR,
    CMP_SEQ_OP_DONE,
    Command,
    Completion,
    D_B_CMD_PREFETCH_BATCH,
    D_B_CMD_RUN_ATTENTION,
    D_B_CMD_RUN_MOE,
    D_CMD_RUN_DSPARK_STEP,
    E_CMD_FETCH_AND_RUN_MOE,
    E_CMD_FETCH_AND_RUN_MOE_BIG,
    E_CMD_REEF_ROUTE,
    E_CMD_FAR_FORWARD_LAYER,
    ExpertPrefetchEntry,
    ExpertEvictionEntry,
    SIDEBAND_EXPERT_EVICTION_OFF,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    RingHeader,
    RoutingExportHeader,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_SPEC_CHECKPOINT_OFF,
    SIDEBAND_TOKEN_IDS_OFF,
)
from bridge.ring_bridge import EngineBridge, GpuLru
from bridge.spec_decode import (PlainAgg, SpecStats, run_plain_loop,
                                run_speculative_loop)

VOCAB = 100_000
NUM_LAYERS = 4
FIRST_MOE = 2
NUM_EXPERTS = 64
NUM_GPUS = 4
TOPK = 4
RING_SLOTS = 64        # small → the 100-token loops exercise wrap-around
SIDEBAND_BYTES = 160_000
SPEC_READBACK_OFF = SIDEBAND_SPEC_CHECKPOINT_OFF + 2560
LOGITS_ROWS = 16       # mirrors ipc::kMaxLogitsReadbackRows


def f(t: int) -> int:
    return (t * 1103515245 + 12345) % VOCAB


def chain(seed: int, n: int) -> list[int]:
    out, t = [], seed
    for _ in range(n):
        t = f(t)
        out.append(t)
    return out


class FakeInfo:
    """EngineInfo shim over a Python-owned IPC block."""

    def __init__(self) -> None:
        hdr = ctypes.sizeof(RingHeader)
        self.cmd_ring_offset = 0
        self.cmd_ring_slots = RING_SLOTS
        self.cmd_slot_bytes = 256
        cmd_sz = hdr + RING_SLOTS * 256
        self.cmp_ring_offset = cmd_sz
        self.cmp_ring_slots = RING_SLOTS
        self.cmp_slot_bytes = 128
        cmp_sz = hdr + RING_SLOTS * 128
        self.sideband_offset = cmd_sz + cmp_sz
        total = self.sideband_offset + SIDEBAND_BYTES
        self._buf = ctypes.create_string_buffer(total)
        self.ipc_base = ctypes.addressof(self._buf)
        for off, slots, sz in ((self.cmd_ring_offset, RING_SLOTS, 256),
                               (self.cmp_ring_offset, RING_SLOTS, 128)):
            rh = RingHeader.from_address(self.ipc_base + off)
            rh.producer_seq = 0
            rh.consumer_seq = 0
            rh.slot_count = slots
            rh.slot_size = sz
        self.num_gpus = NUM_GPUS
        self.num_layers = NUM_LAYERS
        self.num_moe_layers = NUM_LAYERS - FIRST_MOE
        self.num_experts = NUM_EXPERTS
        self.moe_batch_capacity = 512
        # Full-logits readback region (mirrors the engine's pinned host
        # region exposed by pybind logits_readback_addr()): LOGITS_ROWS
        # rows — row 0 = the guided-decoding single-row use, rows [0, R)
        # = the speculative sampled/logprobs verify chunk.
        self._logits_buf = ctypes.create_string_buffer(
            LOGITS_ROWS * VOCAB * 4)
        self.logits_addr = ctypes.addressof(self._logits_buf)
        self.logits_rows = LOGITS_ROWS


class FakeDaemon(threading.Thread):
    """Consume commands, emulate the engine's per-command contracts."""

    def __init__(self, info: FakeInfo, *, conf_enabled: bool,
                 gamma: int, logit_vals: tuple = (1.0, 0.5)) -> None:
        super().__init__(daemon=True)
        self.info = info
        self.conf_enabled = conf_enabled
        self.gamma = gamma
        # Scripted logits row for token t: logit_vals[i] at
        # (f(t) + i) % VOCAB, zeros elsewhere.  The default (1.0, 0.5)
        # is the historical guided-decoding script; sampled-speculation
        # tests pass e.g. (30.0, 29.0, 28.0) so softmax mass concentrates
        # on an analytically tractable 3-token support.
        self.logit_vals = tuple(logit_vals)
        self.stop_flag = False
        self.errors: list[str] = []
        self.dspark_calls = 0
        self.prefetch_batches = 0
        self.reef_routes = 0
        self.far_layers = 0
        base = info.ipc_base
        self._cmd_hdr = base + info.cmd_ring_offset
        self._cmd_slots = self._cmd_hdr + ctypes.sizeof(RingHeader)
        self._cmp_hdr = base + info.cmp_ring_offset
        self._cmp_slots = self._cmp_hdr + ctypes.sizeof(RingHeader)
        self._sideband = base + info.sideband_offset
        self._rows: list[int] = []           # last embedded tokens
        self.known_seqs: set[int] = set()
        self.forks = 0
        self.seq_frees = 0
        # Page-pool admission emulation (evict-at-admission tests): when
        # set, SEQ_CREATE / SEQ_FORK are rejected with the engine's
        # "pool exhausted" CMP_ERROR while len(known_seqs) >= seq_capacity
        # (a live holder pins its "pages" until SEQ_FREE'd — the
        # regression-hunt 2026-08-23 finding (b) shape).
        self.seq_capacity: int | None = None
        self.seq_admission_rejects = 0
        self.embed_calls: list[int] = []     # n per EMBEDDING_LOOKUP
        self.fetch_moes = 0                  # E_CMD_FETCH_AND_RUN_MOE
        self.fetch_moe_bigs = 0              # E_CMD_FETCH_AND_RUN_MOE_BIG
        self.moe_big_rows: list[int] = []    # their num_seqs fields
        self.dense_moes = 0                  # D_B_CMD_RUN_MOE
        self.logits_readbacks = 0            # OUTPUT_HEAD readback_logits=1
        self.samples = 0                     # CMD_SAMPLE_TOKENS
        self.sample_seeds: list[int] = []    # their random_seed fields

    # minimal ring ops (daemon side: cmd consumer, cmp producer)
    def _read_cmd(self) -> Command | None:
        hdr = RingHeader.from_address(self._cmd_hdr)
        cons, prod = hdr.consumer_seq, hdr.producer_seq
        if cons >= prod:
            return None
        src = self._cmd_slots + (cons % RING_SLOTS) * 256
        cmd = Command.from_buffer_copy(ctypes.string_at(src, 256))
        hdr.consumer_seq = cons + 1
        return cmd

    def _write_cmp(self, cmp: Completion) -> None:
        hdr = RingHeader.from_address(self._cmp_hdr)
        prod = hdr.producer_seq
        assert prod - hdr.consumer_seq < RING_SLOTS, "cmp ring full"
        dest = self._cmp_slots + (prod % RING_SLOTS) * 128
        ctypes.memmove(dest, bytes(cmp), 128)
        hdr.producer_seq = prod + 1

    def _done(self, cmd: Command, **compute) -> None:
        c = Completion()
        c.cmp_type = (CMP_SEQ_OP_DONE
                      if cmd.cmd_type in (CMD_SEQ_CREATE, CMD_SEQ_FREE,
                                          CMD_SEQ_FORK)
                      else CMP_COMPUTE_DONE)
        c.cmd_seq = cmd.cmd_seq
        c.gpu_idx = 0
        c.status = 0
        c.payload.compute.cmd_type = cmd.cmd_type
        for k, v in compute.items():
            setattr(c.payload.compute, k, v)
        self._write_cmp(c)

    def run(self) -> None:
        try:
            while not self.stop_flag:
                cmd = self._read_cmd()
                if cmd is None:
                    continue
                self._handle(cmd)
        except Exception as e:  # surfaced by the test after join
            self.errors.append(repr(e))

    def _error(self, cmd: Command, msg: str) -> None:
        c = Completion()
        c.cmp_type = CMP_ERROR
        c.cmd_seq = cmd.cmd_seq
        c.gpu_idx = 0
        c.status = 1
        c.payload.error.error_category = 2
        c.payload.error.message = msg.encode()
        self._write_cmp(c)

    def _admission_full(self) -> bool:
        return (self.seq_capacity is not None
                and len(self.known_seqs) >= self.seq_capacity)

    def _handle(self, cmd: Command) -> None:
        t = cmd.cmd_type
        if t == CMD_SEQ_CREATE:
            if self._admission_full():
                self.seq_admission_rejects += 1
                self._error(cmd, "seq_create: kMain page pool exhausted "
                                 f"(scripted cap {self.seq_capacity})")
                return
            self.known_seqs.add(int(cmd.payload.seq_create.seq_id))
            self._done(cmd)
        elif t == CMD_SEQ_FREE:
            self.known_seqs.discard(int(cmd.payload.seq_free.seq_id))
            self.seq_frees += 1
            self._done(cmd)
        elif t == CMD_SEQ_FORK:
            # Prefix-cache / spec-fork primitive: the fake is stateless per
            # seq (routing derives from the embedded rows), so a fork just
            # validates lifecycle and registers the child.
            if self._admission_full():
                self.seq_admission_rejects += 1
                self._error(cmd, "seq_fork: kMain page pool exhausted "
                                 f"(scripted cap {self.seq_capacity})")
                return
            src = int(cmd.payload.seq_fork.src_seq_id)
            dst = int(cmd.payload.seq_fork.dst_seq_id)
            if src not in self.known_seqs:
                self.errors.append(f"seq_fork: unknown src {src}")
            if dst in self.known_seqs:
                self.errors.append(f"seq_fork: dst {dst} exists")
            self.known_seqs.add(dst)
            self.forks += 1
            self._done(cmd)
        elif t == CMD_EMBEDDING_LOOKUP:
            p = cmd.payload.embedding_lookup
            n = int(p.num_tokens)
            ro = int(p.row_offset)     # SC superchunk staging row
            arr = (ctypes.c_uint32 * n).from_address(
                self._sideband + SIDEBAND_TOKEN_IDS_OFF)
            toks = [int(arr[i]) for i in range(n)]
            # Emulate the engine's hidden staging: rows land at
            # [row_offset, row_offset+n) and persist across sub-chunks.
            if len(self._rows) < ro + n:
                self._rows.extend([0] * (ro + n - len(self._rows)))
            self._rows[ro:ro + n] = toks
            self.embed_calls.append(n)
            self._done(cmd)
        elif t == D_B_CMD_RUN_ATTENTION:
            p = cmd.payload.run_attention
            ro = int(p.row_offset)     # SC sub-launch reads staged rows
            if p.emit_gating:
                hdr = RoutingExportHeader.from_address(
                    self._sideband + SIDEBAND_ROUTING_EXPORT_OFF)
                hdr.num_tokens = p.num_seqs
                hdr.topk = TOPK
                hdr.layer_idx = p.layer_idx
                idx = (ctypes.c_int32 * (p.num_seqs * TOPK)).from_address(
                    self._sideband + SIDEBAND_ROUTING_EXPORT_INDICES_OFF)
                for r in range(p.num_seqs):
                    for j in range(TOPK):
                        idx[r * TOPK + j] = \
                            (self._rows[ro + r] + j) % NUM_EXPERTS
            self._done(cmd, layer_idx=p.layer_idx)
        elif t == E_CMD_FETCH_AND_RUN_MOE:
            p = cmd.payload.fetch_and_run_moe
            if p.expert_count == 0:
                self.errors.append("FETCH_AND_RUN_MOE expert_count 0")
            self.fetch_moes += 1
            self._done(cmd, layer_idx=p.layer_idx)
        elif t == E_CMD_FETCH_AND_RUN_MOE_BIG:
            p = cmd.payload.fetch_and_run_moe_big
            if p.expert_count == 0:
                self.errors.append("FETCH_AND_RUN_MOE_BIG expert_count 0")
            # SC contract: the union must equal the dedup over ALL staged
            # rows' routed sets (the engine consumes the stored gating).
            n = int(p.num_seqs)
            want: set[int] = set()
            for tok in self._rows[:n]:
                for j in range(TOPK):
                    want.add((tok + j) % NUM_EXPERTS)
            pfe = (ExpertPrefetchEntry * int(p.expert_count)).from_address(
                self._sideband + SIDEBAND_EXPERT_PREFETCH_OFF)
            got = {int(pfe[i].expert_idx) for i in range(p.expert_count)}
            if got != want:
                self.errors.append(
                    f"MOE_BIG union mismatch L{p.layer_idx}: "
                    f"got {len(got)} want {len(want)}")
            self.fetch_moe_bigs += 1
            self.moe_big_rows.append(n)
            self._done(cmd, layer_idx=p.layer_idx)
        elif t == E_CMD_REEF_ROUTE:
            # Emulate the daemon ReefOrch service: rewrite gpu targets in
            # place (e%NUM_GPUS stands in for the solver) + sentinel evicts.
            p = cmd.payload.reef_route
            n = int(p.expert_count)
            pfe = (ExpertPrefetchEntry * n).from_address(
                self._sideband + SIDEBAND_EXPERT_PREFETCH_OFF)
            eve = (ExpertEvictionEntry * n).from_address(
                self._sideband + SIDEBAND_EXPERT_EVICTION_OFF)
            for i in range(n):
                pfe[i].gpu_idx = pfe[i].expert_idx % NUM_GPUS
                eve[i].layer_idx = p.layer_idx
                eve[i].expert_idx = 0xFFFF
                eve[i].gpu_idx = pfe[i].gpu_idx
            self.reef_routes += 1
            self._done(cmd, layer_idx=p.layer_idx)
        elif t == E_CMD_FAR_FORWARD_LAYER:
            # Emulate the fused layer: routing union from the embedded rows
            # (the same formula the attention emit_gating path uses here),
            # count via data_bytes; dense layers report 0.
            p = cmd.payload.far_forward_layer
            count = 0
            if p.layer_idx >= FIRST_MOE:
                seen: set[int] = set()
                for tok in self._rows[:p.num_seqs]:
                    for j in range(TOPK):
                        seen.add((tok + j) % NUM_EXPERTS)
                count = len(seen)
            self.far_layers += 1
            self._done(cmd, layer_idx=p.layer_idx, data_bytes=count)
        elif t == D_B_CMD_RUN_MOE:
            self.dense_moes += 1
            self._done(cmd, layer_idx=cmd.payload.run_moe.layer_idx)
        elif t == D_B_CMD_PREFETCH_BATCH:
            self.prefetch_batches += 1
            self._done(cmd)
        elif t == CMD_OUTPUT_HEAD:
            p = cmd.payload.output_head
            if p.readback_logits:
                self.logits_readbacks += 1
                # Engine contract (kMaxLogitsReadbackRows): write the
                # FIRST min(num_tokens, LOGITS_ROWS) rows' full logits.
                # Row b: peak logit_vals[0] at the chain token
                # f(rows[b]), runner-up(s) at the following ids, zeros
                # elsewhere.  num_tokens == 1 is the historical guided
                # single-row script — masked hosts that ban the peak
                # must fall to the runner-up ("grammar repair" token).
                import numpy as np
                n = min(int(p.num_tokens), LOGITS_ROWS)
                arr = np.frombuffer(self.info._logits_buf,
                                    dtype=np.float32).reshape(
                                        LOGITS_ROWS, VOCAB)
                arr[:n] = 0.0
                for b in range(n):
                    peak = f(self._rows[b])
                    for i, v in enumerate(self.logit_vals):
                        arr[b][(peak + i) % VOCAB] = v
            if p.readback_to_host and p.num_tokens > 1:
                arr = (ctypes.c_uint32 * p.num_tokens).from_address(
                    self._sideband + SPEC_READBACK_OFF)
                for b in range(p.num_tokens):
                    arr[b] = f(self._rows[b])
                self._done(cmd, host_buf_offset=SPEC_READBACK_OFF,
                           data_bytes=4 * p.num_tokens,
                           top1_prob=0.9, entropy=0.1)
            else:
                self._done(cmd, top1_prob=0.9, entropy=0.1)
        elif t == CMD_SAMPLE_TOKENS:
            self.samples += 1
            # Record the per-step Philox key the orchestrator derived
            # (the plain sampled arm's correlated-draw fix is asserted
            # against this).  The fake still picks argmax.
            self.sample_seeds.append(int(cmd.payload.sample_tokens
                                         .random_seed))
            arr = (ctypes.c_uint32 * 1).from_address(
                self._sideband + SIDEBAND_TOKEN_IDS_OFF)
            arr[0] = f(self._rows[0])
            self._done(cmd)
        elif t == D_CMD_RUN_DSPARK_STEP:
            p = cmd.payload.run_dspark_step
            g = p.num_query
            # Draft = the true chain with an error injected at a slot
            # cycling per call (g+1 → some rounds are fully correct).
            wrong_at = self.dspark_calls % (g + 1)
            self.dspark_calls += 1
            ids = (ctypes.c_int32 * g).from_address(
                self._sideband + SPEC_READBACK_OFF)
            cf = (ctypes.c_float * g).from_address(
                self._sideband + SPEC_READBACK_OFF + 4 * g)
            tok = p.anchor_token_id
            for k in range(g):
                tok = f(tok)
                ids[k] = tok if k != wrong_at else (tok + 1) % VOCAB
                cf[k] = 0.9 if k != wrong_at else 0.15
            self._done(cmd, host_buf_offset=SPEC_READBACK_OFF,
                       data_bytes=4 * g * (2 if self.conf_enabled else 1),
                       top1_prob=0.9, entropy=0.1)
        else:
            self.errors.append(f"unknown cmd_type 0x{t:x}")


def _make(conf_enabled: bool = False, gamma: int = 5,
          logit_vals: tuple = (1.0, 0.5), logits_rows: int = 1,
          **bridge_kw) -> tuple[EngineBridge, FakeDaemon, FakeInfo]:
    # logits_rows: rows the BRIDGE believes the readback region holds —
    # 1 (default) models the historical single-row engine build (plain
    # fallback routing preserved); LOGITS_ROWS arms the sampled/logprobs
    # speculative arms.
    info = FakeInfo()
    daemon = FakeDaemon(info, conf_enabled=conf_enabled, gamma=gamma,
                        logit_vals=logit_vals)
    bridge = EngineBridge(info, vocab_size=VOCAB,
                          first_moe_layer=FIRST_MOE,
                          hidden_buf_id=7, logits_buf_id=9, **bridge_kw)
    bridge.logits_host_addr = info.logits_addr
    bridge.logits_host_rows = logits_rows
    daemon.start()
    return bridge, daemon, info


def _finish(daemon: FakeDaemon) -> None:
    daemon.stop_flag = True
    daemon.join(timeout=5)
    assert not daemon.errors, daemon.errors


def test_plain_loop_greedy_chain():
    bridge, daemon, _ = _make()
    try:
        agg = PlainAgg()
        out = run_plain_loop(bridge, 1, 30, 4321, None, agg)
        assert out == chain(4321, 30)
        assert agg.lookups > 0 and agg.nan_count == 0
    finally:
        _finish(daemon)


@pytest.mark.parametrize("vb,overlap", [("batched", False),
                                        ("seq", False),
                                        ("batched", True)])
def test_speculative_lossless_chain(vb, overlap):
    bridge, daemon, _ = _make(gamma=5)
    try:
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 40, 4321, None, st,
                                   gamma=5, vb=vb, overlap=overlap)
        n = 40
        assert len(out) >= n
        assert out[:n] == chain(4321, n), f"lossless violated ({vb})"
        assert st.rounds > 0 and st.proposed > 0
        assert 0 < st.accepted < st.proposed  # error injection bites
        assert st.nan_count == 0
        assert daemon.reef_routes > 0  # default arm = REEF over IPC
    finally:
        _finish(daemon)


@pytest.mark.parametrize("route_arm,far_burst", [("reef", True),
                                                 ("act", True),
                                                 ("reef", False)])
def test_far_fused_lossless_chain(route_arm, far_burst):
    """E_CMD_FAR_FORWARD_LAYER path: fused commands per layer — BURST
    (pipelined sliding-window sends over the 64-slot unit rings, forcing
    wrap-around + window back-pressure) and serial forms; the lossless
    invariant and the lookups accumulator (completion data_bytes) must
    hold in every mode."""
    bridge, daemon, _ = _make(gamma=5, use_far=True, route_arm=route_arm,
                              far_burst=far_burst)
    try:
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 40, 4321, None, st,
                                   gamma=5, vb="batched", overlap=True)
        assert out[:40] == chain(4321, 40), "lossless violated (far)"
        assert daemon.far_layers > 0
        assert daemon.reef_routes == 0  # fused path: no separate route cmd
        assert st.lookups > 0           # data_bytes accumulation works
    finally:
        _finish(daemon)


def test_conf_truncation_and_fallback():
    bridge, daemon, _ = _make(conf_enabled=True, gamma=5)
    try:
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 40, 4321, None, st,
                                   gamma=5, conf_thresh=0.5, vb="batched")
        assert out[:40] == chain(4321, 40)
        # wrong_at==0 rounds truncate to g_use=0 → hybrid fallback fires,
        # and the 0.15-conf slot truncates deep slots elsewhere.
        assert st.trunc_slots > 0
        assert st.fallback_rounds > 0
    finally:
        _finish(daemon)


def test_prefetch_records_and_fires():
    bridge, daemon, _ = _make(gamma=5)
    try:
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 40, 4321, None, st,
                                   gamma=5, vb="batched", prefetch=True)
        assert out[:40] == chain(4321, 40)
        assert daemon.prefetch_batches > 0
        assert not bridge.fire_forget_seqs  # all drops consumed
    finally:
        _finish(daemon)


def test_lru_victim_map_chain_neutral():
    bridge, daemon, _ = _make(gamma=5)
    try:
        lrus = [GpuLru(8) for _ in range(NUM_GPUS)]  # tiny → evictions
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 40, 4321, lrus, st,
                                   gamma=5, vb="batched")
        assert out[:40] == chain(4321, 40)
        assert all(len(l.resident) <= l.capacity for l in lrus)
    finally:
        _finish(daemon)


@pytest.mark.skipif(not ring_bridge.fastbridge_active(),
                    reason="_fastbridge not built")
def test_pure_ctypes_fallback_parity(monkeypatch):
    """Same run through the pure-ctypes path must produce the same chain."""
    monkeypatch.setattr(ring_bridge, "_fb", None)
    bridge, daemon, _ = _make(gamma=5)
    try:
        st = SpecStats()
        out = run_speculative_loop(bridge, 1, 30, 999, None, st,
                                   gamma=5, vb="batched", overlap=True)
        assert out[:30] == chain(999, 30)
    finally:
        _finish(daemon)
