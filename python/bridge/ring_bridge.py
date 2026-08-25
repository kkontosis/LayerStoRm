"""EngineBridge — drive the live engine over the raw IPC rings.

The orchestrator-free counterpart of the C++ keeper/dsp52 test fixtures
(tests/integration/dsp52_test.cpp): one class owning the command ring
(producer), completion ring (consumer) and sideband region attached
zero-copy to a live ``layerstorm_engine.start_engine()`` EngineInfo, plus
the per-step command chains those fixtures issue:

  decode_step_fetch_and_run   — B=1 keeper step (EMBED → per-layer
                                RUN_ATTENTION[fused gate] +
                                FETCH_AND_RUN_MOE → OUTPUT_HEAD →
                                SAMPLE_TOKENS argmax)
  verify_step_fetch_and_run   — R-row batched-verify chunk (is_prefill=1,
                                deduped routed union, one OUTPUT_HEAD
                                readback of R argmax ids)
  prefill_chunk_fetch_and_run — prompt prefill chunk (no output head)
  dspark_draft_step / _async  — D_CMD_RUN_DSPARK_STEP + DSP-5/DSP-6
                                sideband readback (ids [+ confidences])

Expert placement is the C++ default arm: static ``expert % num_gpus``
targets plus the 13c-2.0 test-side LRU eviction model (GpuLru victim map,
``have_evict_map=1``) — byte-identical decision logic to
``fill_moe_entries`` in dsp52_test.cpp with REEF-ORCH/affinity off.  The
I8 loader shadow (LS_LOADER_SHADOW=1) runs daemon-side exactly as in the
C++ fixture.

Hot-path acceleration: if the optional Cython module ``bridge._fastbridge``
is importable, ring writes, the completion poll and the bulk sideband
writes go through it; otherwise a pure-ctypes fallback (same semantics,
x86-64 TSO — aligned 64-bit loads/stores are atomic, ctypes acts volatile)
is used.  Set env ``LS_BRIDGE_NO_CYTHON=1`` to force the fallback.
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from dataclasses import dataclass, field

from bridge.protocol import (
    CMD_EMBEDDING_LOOKUP,
    CMD_OUTPUT_HEAD,
    CMD_SAMPLE_TOKENS,
    CMD_SEQ_CREATE,
    CMD_SEQ_FORK,
    CMD_SEQ_FREE,
    CMD_SLOT_BYTES,
    CMP_CHECKPOINT,
    CMP_COMPUTE_DONE,
    CMP_ERROR,
    CMP_SEQ_OP_DONE,
    CMP_SLOT_BYTES,
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
    BatchDescriptorEntry,
    ExpertEvictionEntry,
    ExpertPrefetchEntry,
    MAX_BATCH_DESCRIPTORS,
    MAX_EXPERT_PREFETCH,
    MAX_SIDEBAND_TOKEN_IDS,
    RingHeader,
    RoutingExportHeader,
    SIDEBAND_BATCH_DESCRIPTOR_OFF,
    SIDEBAND_EXPERT_EVICTION_OFF,
    SIDEBAND_EXPERT_PREFETCH_OFF,
    SIDEBAND_ROUTING_EXPORT_INDICES_OFF,
    SIDEBAND_ROUTING_EXPORT_OFF,
    SIDEBAND_TOKEN_IDS_OFF,
)

_RING_HEADER_SIZE = ctypes.sizeof(RingHeader)
_PRODUCER_OFF = RingHeader.producer_seq.offset
_CONSUMER_OFF = RingHeader.consumer_seq.offset

_fb = None
if os.environ.get("LS_BRIDGE_NO_CYTHON") != "1":
    try:
        from bridge import _fastbridge as _fb  # type: ignore[no-redef]
    except ImportError:
        _fb = None


def fastbridge_active() -> bool:
    """True when the Cython hot path is loaded."""
    return _fb is not None


def _fb_v2() -> bool:
    """True when the v2 hot path (direct ring packing + GIL-released wait
    + fused MoE-layer issue) is available."""
    return _fb is not None and getattr(_fb, "API_VERSION", 1) >= 2


class BridgeError(RuntimeError):
    """Fatal IPC error (CMP_ERROR, timeout, or ring overflow)."""


class DsparkDraftError(BridgeError):
    """A DSpark draft step failed — DRAFT-side only, target state intact.

    Raised exclusively by ``dspark_collect_async`` (the per-round draft
    boundary) for (a) a CMP_ERROR carrying the in-flight
    D_CMD_RUN_DSPARK_STEP's own cmd_seq (the runtime declined the step —
    invalid/overflowed drafting context, TD-DSPARK-CTX-CAP class) and
    (b) draft readback parse/validation failures after a successful
    completion.  Drafts are ADVISORY (INV-DSPARK-LOSSLESS): none of these
    touch target KV, so callers may fall back to plain decode for the
    request's remainder (INV-SERVE-SPEC-FALLBACK) instead of failing the
    request.  ``wait()`` STASHES an async-dspark error rather than raising
    it out of a target command's wait — the target completion it was
    waiting for is still in flight and must be consumed normally."""


# ── Lightweight completion view ─────────────────────────────────────────────
# Uniform result of the poll paths (Cython returns the same tuple shape).
# Fields mirror Completion.{header, compute payload}; err_msg only for
# CMP_ERROR.


@dataclass(frozen=True)
class Cmp:
    cmp_type: int
    cmd_seq: int
    gpu_idx: int
    status: int
    cmd_type: int = 0
    layer_idx: int = 0
    host_buf_offset: int = 0
    data_bytes: int = 0
    top1_prob: float = 0.0
    entropy: float = 0.0
    err_msg: str = ""


def _parse_completion(data: bytes) -> Cmp:
    c = Completion.from_buffer_copy(data)
    if c.cmp_type == CMP_ERROR:
        return Cmp(c.cmp_type, c.cmd_seq, c.gpu_idx, c.status,
                   err_msg=bytes(c.payload.error.message).split(b"\0")[0]
                   .decode(errors="replace"))
    p = c.payload.compute
    return Cmp(c.cmp_type, c.cmd_seq, c.gpu_idx, c.status,
               p.cmd_type, p.layer_idx, p.host_buf_offset, p.data_bytes,
               p.top1_prob, p.entropy)


# ── 13c-2.0 test-side per-GPU LRU (GpuLru port, dsp52_test.cpp) ─────────────


class GpuLru:
    """Per-GPU LRU over routed (layer, expert) keys — the 13c-2.0 eviction
    model.  ``order`` front = MRU, back = LRU; ``last_tick`` is a global
    recency stamp shared across GPUs (class-level tick, mirroring the C++
    file-static ``g_lru_tick``)."""

    __slots__ = ("capacity", "order", "resident", "last_tick")
    _tick = 0

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self.order: list[tuple[int, int]] = []       # front = MRU
        self.resident: set[tuple[int, int]] = set()
        self.last_tick: dict[tuple[int, int], int] = {}

    def touch(self, key: tuple[int, int]) -> None:
        try:
            self.order.remove(key)
        except ValueError:
            pass
        self.order.insert(0, key)
        GpuLru._tick += 1
        self.last_tick[key] = GpuLru._tick


@dataclass
class StepTimings:
    embedding_ms: float = 0.0
    output_head_ms: float = 0.0
    sample_ms: float = 0.0
    total_ms: float = 0.0
    attention_ms: list[float] = field(default_factory=list)
    moe_ms: list[float] = field(default_factory=list)


@dataclass
class DecodeResult:
    sampled_token: int = 0
    top1_prob: float = 0.0
    entropy: float = 0.0
    moe_lookups: int = 0
    timings: StepTimings = field(default_factory=StepTimings)


@dataclass
class VerifyChunkResult:
    argmax: list[int] = field(default_factory=list)
    top1_prob: float = 0.0
    entropy: float = 0.0
    moe_lookups: int = 0
    timings: StepTimings = field(default_factory=StepTimings)


class EngineBridge:
    """Rings + sideband + per-step command chains over a live engine."""

    def __init__(self, info, *, vocab_size: int, first_moe_layer: int,
                 hidden_buf_id: int, logits_buf_id: int,
                 route_arm: str = "reef", use_far: bool = False,
                 far_burst: bool = True) -> None:
        # route_arm: "reef" (default — E_CMD_REEF_ROUTE solve + victim map,
        # have_evict_map=1: the C++ champion REEF-arm contract) | "act"
        # (static e%tp targets, no victim map — the legacy bridge arm).
        # use_far: drive layers through the fused E_CMD_FAR_FORWARD_LAYER
        # (one command per layer; route_mode from route_arm).
        self.route_arm = route_arm
        self.use_far = use_far
        # far_burst: pipeline the FAR layer sweep (sliding-window sends,
        # collect-in-order) so Python's re-loop never serializes layers.
        self.far_burst = far_burst
        base = int(info.ipc_base)
        self._cmd_hdr = base + int(info.cmd_ring_offset)
        self._cmd_slots = self._cmd_hdr + _RING_HEADER_SIZE
        self._cmd_count = int(info.cmd_ring_slots)
        self._cmd_mask = self._cmd_count - 1
        self._cmd_slot_bytes = int(info.cmd_slot_bytes)
        assert self._cmd_slot_bytes == CMD_SLOT_BYTES
        self._cmp_hdr = base + int(info.cmp_ring_offset)
        self._cmp_slots = self._cmp_hdr + _RING_HEADER_SIZE
        self._cmp_count = int(info.cmp_ring_slots)
        self._cmp_mask = self._cmp_count - 1
        self._cmp_slot_bytes = int(info.cmp_slot_bytes)
        assert self._cmp_slot_bytes == CMP_SLOT_BYTES
        self.sideband = base + int(info.sideband_offset)

        self.num_gpus = int(info.num_gpus)
        # Ticket J: GPUs that HOST routed experts (static e%moe_gpus
        # placement). Defaults to every engine GPU (GLM EP byte-identical);
        # V4 sets 1 — the second GPU is the dspark draft host and must not
        # receive expert work (an e%2 spread changes the EP-combine bf16
        # rounding and forks the ticket-H golden trajectory).
        self.moe_gpus = int(info.num_gpus)
        self.num_layers = int(info.num_layers)
        self.num_experts = int(info.num_experts)
        self.moe_batch_capacity = int(info.moe_batch_capacity)
        self.vocab_size = int(vocab_size)
        self.first_moe_layer = int(first_moe_layer)
        self.hidden_buf_id = int(hidden_buf_id)
        self.logits_buf_id = int(logits_buf_id)
        # TD-SERVE-NAMED-TOOL-CHOICE: host address of the engine's pinned
        # full-logits readback region (pybind logits_readback_addr()); 0 =
        # unavailable.  Set by the boot path / test harness after
        # construction; consumed by OUTPUT_HEAD readback_logits steps.
        # ``logits_host_rows`` = full [vocab] f32 rows the region holds
        # (engine kMaxLogitsReadbackRows; bytes // (vocab*4)).  1 = the
        # historical single-row build: guided decoding works, the
        # speculative sampled/logprobs arms need >= 2 (verify rows).
        self.logits_host_addr = 0
        self.logits_host_rows = 1
        # Daemon-side expert-fetch deadline for DECODE-shaped steps
        # (FETCH_AND_RUN / FAR timeout_us).  Default = the historical
        # champion value (byte-identical GLM behavior).  V4 boot raises it
        # to the golden harness's 200 s: teacher-forced steps stream
        # experts from the GGUF page cache and a cold layer can exceed
        # 5 s (deepseek_v4_gguf_golden_test.cpp precedent).
        self.decode_timeout_us = 5_000_000

        # v2 hot-path argument tuples (splatted into _fastbridge calls).
        self._cargs = (self._cmd_hdr, self._cmd_slots, self._cmd_mask,
                       self._cmd_count, self._cmd_slot_bytes)
        self._wargs = (self._cmp_hdr, self._cmp_slots, self._cmp_mask,
                       self._cmp_slot_bytes)
        self._v2 = _fb_v2()

        # cmd_seq starts at 1 (C++ drivers reserve-ish 0).
        self._seq = 1
        self.fire_forget_seqs: set[int] = set()
        # DSP52_OVERLAP async draft stash (dspark_send_async/collect).
        self._dspark_pending_seq = 0
        self._dspark_cmp: Cmp | None = None
        # Stashed async-dspark CMP_ERROR message (draft-side failure seen
        # by wait() while servicing a TARGET command) — surfaced as
        # DsparkDraftError at the next dspark_collect_async.
        self._dspark_err: str | None = None
        # Lever-4 prev-chunk-union predictor: layer -> [(l,e,zone,gpu)].
        self.pf_pred: list[list[tuple[int, int, int, int]]] = [
            [] for _ in range(self.num_layers)]

        self._token_ids_addr = self.sideband + SIDEBAND_TOKEN_IDS_OFF
        self._batch_addr = self.sideband + SIDEBAND_BATCH_DESCRIPTOR_OFF
        self._prefetch_addr = self.sideband + SIDEBAND_EXPERT_PREFETCH_OFF
        self._evict_addr = self.sideband + SIDEBAND_EXPERT_EVICTION_OFF
        self._routing_hdr_addr = self.sideband + SIDEBAND_ROUTING_EXPORT_OFF
        self._routing_idx_addr = (self.sideband
                                  + SIDEBAND_ROUTING_EXPORT_INDICES_OFF)

    # ── ring primitives ──────────────────────────────────────────────────

    def _next_seq(self) -> int:
        s = self._seq
        self._seq = s + 1
        return s

    def _cmd(self, cmd_type: int, gpu: int = 0) -> Command:
        c = Command()
        c.cmd_type = cmd_type
        c.cmd_seq = self._next_seq()
        c.gpu_idx = gpu
        c.stream_id = 0
        return c

    def send(self, cmd: Command) -> int:
        """Write one command to the ring; returns its cmd_seq."""
        data = bytes(cmd)
        if _fb is not None:
            ok = _fb.ring_write(self._cmd_hdr, self._cmd_slots,
                                self._cmd_mask, self._cmd_count,
                                self._cmd_slot_bytes, data)
        else:
            hdr = RingHeader.from_address(self._cmd_hdr)
            prod = hdr.producer_seq
            cons = ctypes.c_uint64.from_address(
                self._cmd_hdr + _CONSUMER_OFF).value
            if prod - cons >= self._cmd_count:
                ok = False
            else:
                dest = self._cmd_slots + (prod & self._cmd_mask) \
                    * self._cmd_slot_bytes
                ctypes.memmove(dest, data, self._cmd_slot_bytes)
                hdr.producer_seq = prod + 1
                ok = True
        if not ok:
            raise BridgeError("command ring full")
        return int(cmd.cmd_seq)

    def _poll(self) -> Cmp | None:
        """Read one completion, or None when the ring is empty."""
        if _fb is not None:
            t = _fb.cmp_poll(self._cmp_hdr, self._cmp_slots,
                             self._cmp_mask, self._cmp_slot_bytes)
            if t is None:
                return None
            return Cmp(*t)
        hdr = RingHeader.from_address(self._cmp_hdr)
        cons = hdr.consumer_seq
        prod = ctypes.c_uint64.from_address(
            self._cmp_hdr + _PRODUCER_OFF).value
        if cons >= prod:
            return None
        src = self._cmp_slots + (cons & self._cmp_mask) * self._cmp_slot_bytes
        data = ctypes.string_at(src, self._cmp_slot_bytes)
        hdr.consumer_seq = cons + 1
        return _parse_completion(data)

    def wait(self, expected: int, timeout_s: float = 300.0,
             ctx: str = "") -> Cmp:
        """Wait for one completion of `expected` type.

        Mirrors the C++ fixture's wait(): fire-and-forget completions are
        dropped (errors logged, never fatal), an in-flight async DSpark
        completion is stashed for dspark_collect_async, CMP_CHECKPOINT is
        skipped, CMP_ERROR raises.
        """
        deadline = time.monotonic() + timeout_s
        if self._v2:
            # v2: the spin (incl. checkpoint skip) runs GIL-released in C;
            # only completions of interest surface here.
            while True:
                r = _fb.wait_cmp(*self._wargs, expected,
                                 self._dspark_pending_seq,
                                 deadline - time.monotonic())
                kind = r[0]
                if kind == "ok":
                    # Fire-and-forget completions can share the expected
                    # TYPE — drop by cmd_seq first (C++ wait() order).
                    if (self.fire_forget_seqs
                            and r[1] in self.fire_forget_seqs):
                        self.fire_forget_seqs.discard(r[1])
                        continue
                    return Cmp(expected, r[1], 0, r[2], r[3], r[4], r[5],
                               r[6], r[7], r[8])
                if kind == "err":
                    seq, msg = r[1], r[2]
                    if self.fire_forget_seqs and seq in self.fire_forget_seqs:
                        self.fire_forget_seqs.discard(seq)
                        print(f"  [bridge PF] prefetch error: {msg}",
                              flush=True)
                        continue
                    if seq == self._dspark_pending_seq:
                        # DRAFT-side failure while waiting on a TARGET
                        # command: stash, keep waiting — the expected
                        # completion is still in flight.  Surfaces as
                        # DsparkDraftError at dspark_collect_async.
                        self._dspark_pending_seq = 0
                        self._dspark_err = f"CMP_ERROR (async dspark): {msg}"
                        continue
                    raise BridgeError(
                        f"CMP_ERROR{' (' + ctx + ')' if ctx else ''}: {msg}")
                if kind == "dspark":
                    self._dspark_cmp = Cmp(*r[1])
                    self._dspark_pending_seq = 0
                    continue
                if kind == "other":
                    seq = r[1][1]
                    if self.fire_forget_seqs and seq in self.fire_forget_seqs:
                        self.fire_forget_seqs.discard(seq)
                    continue
                raise BridgeError(
                    f"timeout waiting for cmp 0x{expected:x}"
                    f"{' (' + ctx + ')' if ctx else ''}")
        while time.monotonic() < deadline:
            out = self._poll()
            if out is None:
                os.sched_yield()
                continue
            if self.fire_forget_seqs and out.cmd_seq in self.fire_forget_seqs:
                self.fire_forget_seqs.discard(out.cmd_seq)
                if out.cmp_type == CMP_ERROR:
                    print(f"  [bridge PF] prefetch error: {out.err_msg}",
                          flush=True)
                continue
            if (self._dspark_pending_seq
                    and out.cmd_seq == self._dspark_pending_seq):
                if out.cmp_type == CMP_ERROR:
                    # Draft-side failure — stash, keep waiting (see the
                    # v2 arm above / DsparkDraftError).
                    self._dspark_pending_seq = 0
                    self._dspark_err = (f"CMP_ERROR (async dspark): "
                                        f"{out.err_msg}")
                    continue
                self._dspark_cmp = out
                self._dspark_pending_seq = 0
                continue
            if out.cmp_type == CMP_ERROR:
                raise BridgeError(
                    f"CMP_ERROR{' (' + ctx + ')' if ctx else ''}: "
                    f"{out.err_msg}")
            if out.cmp_type == CMP_CHECKPOINT:
                continue
            if out.cmp_type == expected:
                return out
        raise BridgeError(
            f"timeout waiting for cmp 0x{expected:x}"
            f"{' (' + ctx + ')' if ctx else ''}")

    # ── sideband writes/reads ────────────────────────────────────────────

    def write_token_ids(self, toks: list[int]) -> None:
        n = len(toks)
        assert n <= MAX_SIDEBAND_TOKEN_IDS
        if _fb is not None:
            _fb.write_u32(self._token_ids_addr, toks)
            return
        arr = (ctypes.c_uint32 * n).from_address(self._token_ids_addr)
        for i, t in enumerate(toks):
            arr[i] = t

    def write_batch_descriptors(self, seq_id: int, pos0: int,
                                n: int) -> None:
        """n rows of ONE sequence at contiguous positions pos0+b."""
        if _fb is not None:
            _fb.write_batch_desc(self._batch_addr, seq_id, pos0, n)
            return
        arr = (BatchDescriptorEntry * n).from_address(self._batch_addr)
        for b in range(n):
            arr[b].seq_id = seq_id
            arr[b].token_pos = pos0 + b
            arr[b]._pad = 0

    def read_routing_union(self, layer: int, expect_rows: int) -> list[int]:
        """Read the fused-gate routing export and return the DEDUPED
        routed-expert union in first-occurrence (selection-rank) order.

        Asserts row-count and layer identity like the C++ fixture."""
        hdr = RoutingExportHeader.from_address(self._routing_hdr_addr)
        if int(hdr.num_tokens) != expect_rows:
            raise BridgeError(
                f"routing export rows {hdr.num_tokens} != {expect_rows} "
                f"(L{layer})")
        if int(hdr.layer_idx) != layer:
            raise BridgeError(
                f"routing-export layer mismatch: {hdr.layer_idx} != {layer}")
        rn = int(hdr.num_tokens) * int(hdr.topk)
        if _fb is not None:
            return _fb.routing_union(self._routing_idx_addr, rn,
                                     self.num_experts)
        arr = (ctypes.c_int32 * rn).from_address(self._routing_idx_addr)
        seen = bytearray(self.num_experts)
        out: list[int] = []
        for k in range(rn):
            e = arr[k]
            if e < 0 or e >= self.num_experts or seen[e]:
                continue
            seen[e] = 1
            out.append(e)
        return out

    def _write_moe_entries(self, layer: int,
                           experts: list[int], assigns: list[int],
                           victims: list[tuple[int, int, int]]) -> int:
        """Write ExpertPrefetchEntry[] + index-aligned ExpertEvictionEntry[].

        victims[i] = (layer, expert, gpu) or (layer, 0xFFFF, gpu) sentinel.
        """
        n = len(experts)
        assert n <= MAX_EXPERT_PREFETCH
        if _fb is not None:
            _fb.write_moe_entries(self._prefetch_addr, self._evict_addr,
                                  layer, experts, assigns, victims)
            return n
        pfe = (ExpertPrefetchEntry * n).from_address(self._prefetch_addr)
        eve = (ExpertEvictionEntry * n).from_address(self._evict_addr)
        for i in range(n):
            pfe[i].layer_idx = layer
            pfe[i].expert_idx = experts[i]
            pfe[i].zone = 0
            pfe[i].gpu_idx = assigns[i]
            vl, ve, vg = victims[i]
            eve[i].layer_idx = vl
            eve[i].expert_idx = ve
            eve[i].gpu_idx = vg
            eve[i]._pad = 0
        return n

    def read_spec_readback_ids(self, host_buf_offset: int, gamma: int,
                               with_conf: bool
                               ) -> tuple[list[int], list[float]]:
        """DSP-5/DSP-6 sideband readback: gamma i32 draft ids
        [+ gamma f32 raw survival confidences]."""
        addr = self.sideband + host_buf_offset
        ids_arr = (ctypes.c_int32 * gamma).from_address(addr)
        ids = [int(ids_arr[k]) for k in range(gamma)]
        confs: list[float] = []
        if with_conf:
            cf = (ctypes.c_float * gamma).from_address(addr + 4 * gamma)
            confs = [float(cf[k]) for k in range(gamma)]
        return ids, confs

    def read_head_readback_ids(self, host_buf_offset: int,
                               n: int) -> list[int]:
        """Batched-verify OUTPUT_HEAD readback: n u32 argmax ids."""
        arr = (ctypes.c_uint32 * n).from_address(
            self.sideband + host_buf_offset)
        return [int(arr[b]) for b in range(n)]

    def read_sampled_token(self) -> int:
        return int(ctypes.c_uint32.from_address(self._token_ids_addr).value)

    # ── sequence lifecycle ───────────────────────────────────────────────

    def create_sequence(self, seq_id: int, prompt_len: int) -> None:
        if self._v2:
            if not _fb.send_seq_create(*self._cargs, self._next_seq(),
                                       seq_id, prompt_len):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_SEQ_CREATE)
            c.payload.seq_create.seq_id = seq_id
            c.payload.seq_create.prompt_len = prompt_len
            c.payload.seq_create.pool = 0
            self.send(c)
        out = self.wait(CMP_SEQ_OP_DONE, ctx=f"seq_create {seq_id}")
        if out.status != 0:
            raise BridgeError(f"seq_create {seq_id} status {out.status}")

    def free_sequence(self, seq_id: int) -> None:
        if self._v2:
            if not _fb.send_seq_free(*self._cargs, self._next_seq(), seq_id):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_SEQ_FREE)
            c.payload.seq_free.seq_id = seq_id
            self.send(c)
        try:
            self.wait(CMP_SEQ_OP_DONE, timeout_s=60.0,
                      ctx=f"seq_free {seq_id}")
        except BridgeError as e:      # teardown best-effort (C++ parity)
            print(f"  [bridge] seq_free {seq_id}: {e}", flush=True)

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> None:
        """CMD_SEQ_FORK: CoW-fork src's KV (+ DSA indexer-K) pages into
        dst — the prefix-cache / speculative-fork primitive. Raises on
        pool exhaustion (caller may evict and retry)."""
        if self._v2:
            if not _fb.send_seq_fork(*self._cargs, self._next_seq(),
                                     src_seq_id, dst_seq_id):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_SEQ_FORK)
            c.payload.seq_fork.src_seq_id = src_seq_id
            c.payload.seq_fork.dst_seq_id = dst_seq_id
            self.send(c)
        out = self.wait(CMP_SEQ_OP_DONE,
                        ctx=f"seq_fork {src_seq_id}->{dst_seq_id}")
        if out.status != 0:
            raise BridgeError(
                f"seq_fork {src_seq_id}->{dst_seq_id} status {out.status}")

    # ── expert placement + 13c-2.0 LRU eviction map ──────────────────────

    def fill_moe_entries(self, layer: int, topk: list[int],
                         lrus: list[GpuLru] | None) -> int:
        """Static e%tp placement + the 13c-2.0 LRU victim map — the
        fill_moe_entries default arm (REEF-ORCH/affinity off) of
        dsp52_test.cpp, byte-identical decision logic."""
        tp = self.moe_gpus
        n = len(topk)
        assigns = [e % tp for e in topk]
        victims: list[tuple[int, int, int]] = [
            (layer, 0xFFFF, assigns[i]) for i in range(n)]

        if lrus is not None:
            for g in range(tp):
                lru = lrus[g]
                want: list[int] = []
                miss_ei: list[int] = []
                for i in range(n):
                    if assigns[i] != g:
                        continue
                    e = topk[i]
                    want.append(e)
                    k = (layer, e)
                    if k in lru.resident:
                        lru.touch(k)
                    else:
                        miss_ei.append(i)
                want_set = set(want)

                def needed_now(k: tuple[int, int]) -> bool:
                    return k[0] == layer and k[1] in want_set

                need_room = len(miss_ei)
                vi = 0
                while (len(lru.resident) + need_room > lru.capacity
                       and vi < len(miss_ei)):
                    victim = None
                    for k in reversed(lru.order):
                        if not needed_now(k):
                            victim = k
                            break
                    if victim is None:
                        break
                    ei = miss_ei[vi]
                    vi += 1
                    victims[ei] = (victim[0], victim[1], g)
                    lru.resident.discard(victim)
                    lru.last_tick.pop(victim, None)
                    try:
                        lru.order.remove(victim)
                    except ValueError:
                        pass
                for ei in miss_ei:
                    k = (layer, topk[ei])
                    lru.resident.add(k)
                    lru.touch(k)

        return self._write_moe_entries(layer, topk, assigns, victims)

    # ── per-step command chains ──────────────────────────────────────────

    def _run_attention(self, layer: int, num_seqs: int, *, is_prefill: int,
                       chunk_start: int, chunk_len: int,
                       is_moe: bool, superchunk: int = 0,
                       row_offset: int = 0) -> None:
        if self._v2:
            if not _fb.send_attention(*self._cargs, self._next_seq(),
                                      layer, num_seqs, is_prefill,
                                      chunk_start, chunk_len,
                                      1 if is_moe else 0,
                                      superchunk, row_offset):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(D_B_CMD_RUN_ATTENTION)
            p = c.payload.run_attention
            p.layer_idx = layer
            p.num_seqs = num_seqs
            p.is_prefill = is_prefill
            p.use_graph = 0
            p.is_draft = 0
            p.emit_checkpoint = 0
            p.chunk_start = chunk_start
            p.chunk_len = chunk_len
            p.emit_gating = 1 if is_moe else 0
            p.store_gating = 1 if is_moe else 0
            p.superchunk = superchunk
            p.row_offset = row_offset
            self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"attn L{layer}")
        if out.status != 0:
            raise BridgeError(f"attn L{layer} status {out.status}")

    def _run_dense_moe(self, layer: int, num_seqs: int) -> None:
        if self._v2:
            if not _fb.send_dense_moe(*self._cargs, self._next_seq(),
                                      layer, num_seqs):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(D_B_CMD_RUN_MOE)
            p = c.payload.run_moe
            p.layer_idx = layer
            p.num_seqs = num_seqs
            p.moe_mode = 0
            p.apply_residual_correction = 0
            p.store_gating_output = 0
            p.emit_checkpoint = 0
            self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"dense moe L{layer}")
        if out.status != 0:
            raise BridgeError(f"dense moe L{layer} status {out.status}")

    def _reef_moe(self, layer: int, num_seqs: int, timeout_us: int) -> int:
        """REEF-arm routed-MoE layer: the union goes to the daemon's
        ReefOrch service via E_CMD_REEF_ROUTE (it rewrites the sideband
        gpu targets + fills the 13c-2.0 victim map), then
        E_CMD_FETCH_AND_RUN_MOE ships with have_evict_map=1 — the C++
        champion REEF-arm contract. Ring order lets both commands publish
        back-to-back; two completions are collected."""
        if self._v2:
            n = _fb.reef_route_fetch_from_export(
                *self._cargs, self._next_seq(), self._next_seq(),
                self._routing_hdr_addr, self._routing_idx_addr,
                self._prefetch_addr, layer, num_seqs, self.num_experts,
                timeout_us, MAX_EXPERT_PREFETCH)
            if n == -1:
                raise BridgeError("command ring full")
            if n == -2 or n == -3:
                raise BridgeError(
                    f"routing-export mismatch in REEF_ROUTE (L{layer})")
            if n == -4:
                raise BridgeError(f"no routed experts exported L{layer}")
            out = self.wait(CMP_COMPUTE_DONE, ctx=f"reef route L{layer}")
            if out.status != 0:
                raise BridgeError(f"reef route L{layer} status {out.status}")
            out = self.wait(CMP_COMPUTE_DONE, ctx=f"moe L{layer}")
            if out.status != 0:
                raise BridgeError(f"moe L{layer} status {out.status}")
            return n
        topk = self.read_routing_union(layer, num_seqs)
        if not topk:
            raise BridgeError(f"no routed experts exported L{layer}")
        n = self._write_moe_entries(
            layer, topk, [0] * len(topk),
            [(layer, 0xFFFF, 0)] * len(topk))
        c = self._cmd(E_CMD_REEF_ROUTE)
        c.payload.reef_route.layer_idx = layer
        c.payload.reef_route.expert_count = n
        self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"reef route L{layer}")
        if out.status != 0:
            raise BridgeError(f"reef route L{layer} status {out.status}")
        self._fetch_and_run_moe(layer, num_seqs, n, have_evict_map=True,
                                timeout_us=timeout_us)
        return n

    def _send_far_cmd(self, layer: int, num_seqs: int, *, is_prefill: int,
                      chunk_start: int, chunk_len: int,
                      timeout_us: int) -> None:
        mode = 1 if self.route_arm == "reef" else 0
        if self._v2:
            if not _fb.send_far_layer(*self._cargs, self._next_seq(),
                                      layer, num_seqs, chunk_start,
                                      chunk_len, timeout_us, is_prefill,
                                      mode):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(E_CMD_FAR_FORWARD_LAYER)
            p = c.payload.far_forward_layer
            p.layer_idx = layer
            p.num_seqs = num_seqs
            p.chunk_start = chunk_start
            p.chunk_len = chunk_len
            p.timeout_us = timeout_us
            p.is_prefill = is_prefill
            p.route_mode = mode
            self.send(c)

    def _far_layer(self, layer: int, num_seqs: int, *, is_prefill: int,
                   chunk_start: int, chunk_len: int,
                   timeout_us: int) -> int:
        """One fused layer via E_CMD_FAR_FORWARD_LAYER (attention + fused
        gate + routed FETCH_AND_RUN, or the dense path, daemon-side).
        Returns the deduped entry count (completion data_bytes; 0 dense)."""
        self._send_far_cmd(layer, num_seqs, is_prefill=is_prefill,
                           chunk_start=chunk_start, chunk_len=chunk_len,
                           timeout_us=timeout_us)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"far L{layer}")
        if out.status != 0:
            raise BridgeError(f"far L{layer} status {out.status}")
        return out.data_bytes

    def _far_sweep_burst(self, num_seqs: int, *, is_prefill: int,
                         chunk_start: int, chunk_len: int, timeout_us: int,
                         moe_ms: list | None = None) -> int:
        """FAR-arm layer sweep with PIPELINED sends (sliding in-flight
        window): publish FAR commands for all layers back-to-back so the
        daemon executes the sweep in ring order with NO Python turnaround
        on the critical path — Python's inter-command re-loop cost
        (interpreter frames, Cmp build, bookkeeping) is paid concurrently
        with daemon work instead of serializing it (TD-BRIDGE-CPP-GAP).

        The window is capped to half the smaller ring so the cmd ring
        cannot fill and undrained completions cannot overrun the cmp ring
        (dspark-stash/fire-forget slots included in the margin). Per-layer
        walls are recovered as completion ARRIVAL DELTAS (the daemon is
        serial, so arrival spacing == layer duration; the first layer
        absorbs the pipeline fill). Returns Σ data_bytes (lookups)."""
        window = max(4, min(self._cmd_count, self._cmp_count) // 2)
        lookups = 0
        pending: list[int] = []   # layer ids in flight (FIFO)
        head = 0
        t_prev = time.monotonic()
        first_err: BridgeError | None = None

        def collect_one() -> None:
            # DRAIN-BEFORE-RAISE (TD-INDEXER-POOL-EVICT): with sends
            # PIPELINED, a failing layer still leaves the rest of the sweep
            # in flight. Propagating immediately would abandon their
            # completions in the cmp ring, and the NEXT command would read
            # one of THEM — the failure would migrate to an unrelated
            # command (observed: a stale far CMP_ERROR resurfacing as
            # "seq_free N: CMP_ERROR"). Stash the first failure, keep
            # collecting until the ring is clean, then raise it. This is
            # what makes a RETRYABLE engine error (pool exhaustion) safe to
            # answer with an eviction + re-issue of the identical chunk.
            nonlocal lookups, head, t_prev, first_err
            layer = pending[head]
            head += 1
            try:
                out = self.wait(CMP_COMPUTE_DONE, ctx=f"far L{layer} (burst)")
            except BridgeError as err:
                if first_err is None:
                    first_err = err
                return
            if out.status != 0:
                if first_err is None:
                    first_err = BridgeError(
                        f"far L{layer} status {out.status}")
                return
            lookups += out.data_bytes
            now = time.monotonic()
            if moe_ms is not None:
                moe_ms[layer] = (now - t_prev) * 1e3
            t_prev = now

        for layer in range(self.num_layers):
            while len(pending) - head >= window:
                collect_one()
            if first_err is not None:
                break            # stop sending; drain what is already out
            self._send_far_cmd(layer, num_seqs, is_prefill=is_prefill,
                               chunk_start=chunk_start,
                               chunk_len=chunk_len, timeout_us=timeout_us)
            pending.append(layer)
        while head < len(pending):
            collect_one()
        if first_err is not None:
            raise first_err
        return lookups

    def _routed_moe(self, layer: int, num_seqs: int, lrus, *,
                    timeout_us: int, prefetch_record: bool = False) -> int:
        """One routed-MoE layer after its gated attention: consume the
        routing export and issue the routed-MoE command chain; returns the
        deduped entry count (Σ over the run = the keeper `lookups`).

        REEF arm (default, lrus None): E_CMD_REEF_ROUTE + FETCH with the
        victim map (_reef_moe). ACT arm: static e%tp targets, no victim
        map — one fused C call on v2. LRU arm (lrus given) / lever-4
        prefetch recording: the original read→fill→send chain.
        """
        if lrus is None and not prefetch_record and self.route_arm == "reef":
            return self._reef_moe(layer, num_seqs, timeout_us)
        if lrus is None and self._v2 and not prefetch_record:
            n = _fb.fetch_moe_from_export(
                *self._cargs, self._next_seq(),
                self._routing_hdr_addr, self._routing_idx_addr,
                self._prefetch_addr, self._evict_addr,
                layer, num_seqs, self.num_experts, self.moe_gpus,
                timeout_us, MAX_EXPERT_PREFETCH)
            if n == -1:
                raise BridgeError("command ring full")
            if n == -2 or n == -3:
                raise BridgeError(
                    f"routing-export mismatch in FETCH_AND_RUN (L{layer})")
            if n == -4:
                raise BridgeError(f"no routed experts exported L{layer}")
            count = n
        else:
            topk = self.read_routing_union(layer, num_seqs)
            if not topk:
                raise BridgeError(f"no routed experts exported L{layer}")
            count = self.fill_moe_entries(layer, topk, lrus)
            if prefetch_record:
                self.pf_pred[layer] = [
                    (layer, e, 0, e % self.moe_gpus) for e in topk]
            self._fetch_and_run_moe(layer, num_seqs, count,
                                    have_evict_map=lrus is not None,
                                    timeout_us=timeout_us)
            return count
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"moe L{layer}")
        if out.status != 0:
            raise BridgeError(f"moe L{layer} status {out.status}")
        return count

    def _fetch_and_run_moe(self, layer: int, num_seqs: int, count: int,
                           have_evict_map: bool, timeout_us: int) -> None:
        c = self._cmd(E_CMD_FETCH_AND_RUN_MOE)
        p = c.payload.fetch_and_run_moe
        p.layer_idx = layer
        p.num_seqs = num_seqs
        p.expert_count = count
        p.timeout_us = timeout_us
        p.moe_mode = 0
        p.have_evict_map = 1 if have_evict_map else 0
        self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"moe L{layer}")
        if out.status != 0:
            raise BridgeError(f"moe L{layer} status {out.status}")

    def _embed(self, toks: list[int], row_offset: int = 0) -> None:
        self.write_token_ids(toks)
        if self._v2:
            if not _fb.send_embed(*self._cargs, self._next_seq(),
                                  len(toks), self.hidden_buf_id,
                                  row_offset):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_EMBEDDING_LOOKUP)
            p = c.payload.embedding_lookup
            p.num_tokens = len(toks)
            p.output_buf_id = self.hidden_buf_id
            p.row_offset = row_offset
            self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx="embedding")
        if out.status != 0:
            raise BridgeError(f"embedding status {out.status}")

    def _fetch_and_run_moe_big(self, layer: int, num_seqs: int, count: int,
                               timeout_us: int,
                               have_evict_map: bool = False) -> None:
        """SC (superchunk port): ONE big-batch fetch + chunked MoE over the
        whole superchunk (E_CMD_FETCH_AND_RUN_MOE_BIG; num_seqs may exceed
        MAX_BATCH_DESCRIPTORS up to EngineInfo.moe_batch_capacity). Low
        frequency (one per layer per superchunk) — ctypes packing, no .pyx
        fast path needed.  ``have_evict_map``: the sideband eviction map
        was filled (REEF-armed GLM serving superchunk — the 13c-2.0 victim
        map from E_CMD_REEF_ROUTE)."""
        c = self._cmd(E_CMD_FETCH_AND_RUN_MOE_BIG)
        p = c.payload.fetch_and_run_moe_big
        p.layer_idx = layer
        p.num_seqs = num_seqs
        p.expert_count = count
        p.timeout_us = timeout_us
        p.moe_mode = 0
        p.have_evict_map = 1 if have_evict_map else 0
        p.chunk_tokens = 0            # engine default chunk size
        self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx=f"moe-big L{layer}")
        if out.status != 0:
            raise BridgeError(f"moe-big L{layer} status {out.status}")

    def _output_head(self, num_tokens: int, *, readback: bool,
                     readback_logits: bool = False) -> Cmp:
        # readback_logits (guided decoding) rides the ctypes packing even
        # when the v2 fast path is loaded: constrained steps are rare and
        # never wall-critical, so the .pyx stays untouched (the champion
        # fast path is byte-identical for unconstrained requests).
        if self._v2 and not readback_logits:
            if not _fb.send_head(*self._cargs, self._next_seq(), num_tokens,
                                 self.hidden_buf_id, self.logits_buf_id,
                                 1 if readback else 0):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_OUTPUT_HEAD)
            p = c.payload.output_head
            p.num_tokens = num_tokens
            p.input_buf_id = self.hidden_buf_id
            p.output_buf_id = self.logits_buf_id
            p.readback_to_host = 1 if readback else 0
            p.compute_confidence = 1
            p.num_logprobs = 0
            p.mtp_head = 0
            p.readback_logits = 1 if readback_logits else 0
            self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx="output_head")
        if out.status != 0:
            raise BridgeError(f"output_head status {out.status}")
        return out

    def _sample(self, num_tokens: int, temperature: float = 0.0,
                top_p: float = 1.0, top_k: int = 0,
                seed: int = 42) -> None:
        """CMD_SAMPLE_TOKENS. Defaults = the champion greedy arm (argmax,
        byte-identical to the historical form)."""
        if self._v2:
            if not _fb.send_sample(*self._cargs, self._next_seq(),
                                   num_tokens, self.logits_buf_id,
                                   self.vocab_size, temperature, top_p,
                                   top_k, seed):
                raise BridgeError("command ring full")
        else:
            c = self._cmd(CMD_SAMPLE_TOKENS)
            p = c.payload.sample_tokens
            p.num_tokens = num_tokens
            p.logits_buf_id = self.logits_buf_id
            p.vocab_size = self.vocab_size
            p.temperature = temperature
            p.top_p = top_p
            p.top_k = top_k
            p.random_seed = seed
            self.send(c)
        out = self.wait(CMP_COMPUTE_DONE, ctx="sample")
        if out.status != 0:
            raise BridgeError(f"sample status {out.status}")

    def _sample_argmax(self, num_tokens: int) -> None:
        self._sample(num_tokens)

    def _prefetch_fire(self, entries: list[tuple[int, int, int, int]]
                       ) -> None:
        """Lever-4 fire-and-forget prev-chunk-union prefetch (priority -1,
        below the FED's 0.0). Entries are (layer, expert, zone, gpu)."""
        n = min(len(entries), MAX_EXPERT_PREFETCH)
        if n == 0:
            return
        pfe = (ExpertPrefetchEntry * n).from_address(self._prefetch_addr)
        for i in range(n):
            layer, e, zone, gpu = entries[i]
            pfe[i].layer_idx = layer
            pfe[i].expert_idx = e
            pfe[i].zone = zone
            pfe[i].gpu_idx = gpu
        c = self._cmd(D_B_CMD_PREFETCH_BATCH)
        p = c.payload.prefetch_batch
        p.count = n
        p.priority = -1.0
        p.delay_us = 0
        self.fire_forget_seqs.add(int(c.cmd_seq))
        self.send(c)

    def _decode_layers_seq(self, r, lrus) -> None:
        """Sequential (non-burst) decode layer loop: one send+wait per
        command — the FAR-per-layer, split-REEF, ACT and LRU arms."""
        for layer in range(self.num_layers):
            is_moe = layer >= self.first_moe_layer
            if self.use_far and lrus is None:
                tm = time.monotonic()
                r.moe_lookups += self._far_layer(
                    layer, 1, is_prefill=0, chunk_start=0, chunk_len=0,
                    timeout_us=self.decode_timeout_us)
                r.timings.moe_ms[layer] = (time.monotonic() - tm) * 1e3
                continue
            ta = time.monotonic()
            self._run_attention(layer, 1, is_prefill=0, chunk_start=0,
                                chunk_len=0, is_moe=is_moe)
            r.timings.attention_ms[layer] = (time.monotonic() - ta) * 1e3
            tm = time.monotonic()
            if is_moe:
                r.moe_lookups += self._routed_moe(
                    layer, 1, lrus, timeout_us=self.decode_timeout_us)
            else:
                self._run_dense_moe(layer, 1)
            r.timings.moe_ms[layer] = (time.monotonic() - tm) * 1e3

    def decode_step_fetch_and_run(self, input_token: int, seq_id: int,
                                  token_pos: int,
                                  lrus: list[GpuLru] | None,
                                  sampling: tuple | None = None,
                                  logits_readback: bool = False,
                                  logprobs_readback: bool = False
                                  ) -> DecodeResult:
        """One plain B=1 decode step (keeper chain).

        ``sampling`` = (temperature, top_p, top_k, seed) for the sampled
        serving path; None = argmax (byte-identical champion arm).

        ``logits_readback`` (guided decoding): OUTPUT_HEAD carries
        readback_logits=1 — the daemon D2H-copies the row's full logits to
        the pinned host row at ``logits_host_addr`` — and CMD_SAMPLE_TOKENS
        is skipped entirely: the caller owns the (masked) sampling and
        ``sampled_token`` is -1.

        ``logprobs_readback`` (TD-ORCH-LOGPROBS serving): OUTPUT_HEAD
        carries readback_logits=1 like ``logits_readback``, but the
        sampling command still runs ENGINE-side — the token pick stays
        byte-identical to a logprobs-off step while the caller computes
        log-softmax/top-K host-side from the pinned logits row (valid once
        the head completion fired; the sample reads the DEVICE logits
        buffer, so the D2H copy cannot race it)."""
        t_start = time.monotonic()
        r = DecodeResult()
        r.timings.attention_ms = [0.0] * self.num_layers
        r.timings.moe_ms = [0.0] * self.num_layers

        t0 = time.monotonic()
        self._embed([input_token])
        r.timings.embedding_ms = (time.monotonic() - t0) * 1e3

        self.write_batch_descriptors(seq_id, token_pos, 1)

        if self.use_far and lrus is None and self.far_burst:
            r.moe_lookups += self._far_sweep_burst(
                1, is_prefill=0, chunk_start=0, chunk_len=0,
                timeout_us=self.decode_timeout_us, moe_ms=r.timings.moe_ms)
        else:
            self._decode_layers_seq(r, lrus)

        to = time.monotonic()
        out = self._output_head(1, readback=False,
                                readback_logits=(logits_readback
                                                 or logprobs_readback))
        r.top1_prob = out.top1_prob
        r.entropy = out.entropy
        r.timings.output_head_ms = (time.monotonic() - to) * 1e3

        if logits_readback:
            # Guided decoding: the caller masks + samples host-side from
            # the pinned logits row; no CMD_SAMPLE_TOKENS in the chain.
            r.sampled_token = -1
            r.timings.total_ms = (time.monotonic() - t_start) * 1e3
            return r

        ts = time.monotonic()
        if sampling is None:
            self._sample_argmax(1)
        else:
            self._sample(1, *sampling)
        r.timings.sample_ms = (time.monotonic() - ts) * 1e3

        r.sampled_token = self.read_sampled_token()
        r.timings.total_ms = (time.monotonic() - t_start) * 1e3
        return r

    def verify_step_fetch_and_run(self, toks: list[int], seq_id: int,
                                  pos0: int, lrus: list[GpuLru] | None,
                                  *, prefetch: bool = False,
                                  logits_readback: bool = False
                                  ) -> VerifyChunkResult:
        """One R-row batched-verify chunk (dsp52 shape).

        ``logits_readback`` (TD-ORCH-SAMPLED-SPEC / TD-ORCH-LOGPROBS-SPEC):
        OUTPUT_HEAD additionally carries readback_logits=1 — the daemon
        D2H-copies the FIRST min(R, logits_host_rows) rows' full logits
        into the pinned region at ``logits_host_addr`` (host-visible once
        the head completion fired).  The engine-side argmax readback is
        unchanged, so the default arm stays byte-identical."""
        t_start = time.monotonic()
        r = VerifyChunkResult()
        r.timings.attention_ms = [0.0] * self.num_layers
        r.timings.moe_ms = [0.0] * self.num_layers
        R = len(toks)

        t0 = time.monotonic()
        self._embed(toks)
        r.timings.embedding_ms = (time.monotonic() - t0) * 1e3

        self.write_batch_descriptors(seq_id, pos0, R)

        burst = (self.use_far and lrus is None and not prefetch
                 and self.far_burst)
        if burst:
            r.moe_lookups += self._far_sweep_burst(
                R, is_prefill=1, chunk_start=pos0, chunk_len=R,
                timeout_us=self.decode_timeout_us,
                moe_ms=r.timings.moe_ms)
        for layer in ([] if burst else range(self.num_layers)):
            is_moe = layer >= self.first_moe_layer

            # Lever 4 (DSP52_PREFETCH): previous chunk's layer-L union at
            # attention-launch (H2D-idle window); the sideband prefetch
            # region is consumed by the daemon before the attention
            # completion below (ring order), so the fill_moe_entries
            # overwrite is race-free.
            if prefetch and is_moe and self.pf_pred[layer]:
                self._prefetch_fire(self.pf_pred[layer])

            if self.use_far and lrus is None and not prefetch:
                tm = time.monotonic()
                r.moe_lookups += self._far_layer(
                    layer, R, is_prefill=1, chunk_start=pos0, chunk_len=R,
                    timeout_us=self.decode_timeout_us)
                r.timings.moe_ms[layer] = (time.monotonic() - tm) * 1e3
                continue
            ta = time.monotonic()
            self._run_attention(layer, R, is_prefill=1, chunk_start=pos0,
                                chunk_len=R, is_moe=is_moe)
            r.timings.attention_ms[layer] = (time.monotonic() - ta) * 1e3
            tm = time.monotonic()
            if is_moe:
                # Lever-4 recording needs the topk list — the fused path
                # skips it, so prefetch routes through the fallback chain.
                r.moe_lookups += self._routed_moe(
                    layer, R, lrus, timeout_us=self.decode_timeout_us,
                    prefetch_record=prefetch)
            else:
                self._run_dense_moe(layer, R)
            r.timings.moe_ms[layer] = (time.monotonic() - tm) * 1e3

        to = time.monotonic()
        out = self._output_head(R, readback=True,
                                readback_logits=logits_readback)
        r.top1_prob = out.top1_prob
        r.entropy = out.entropy
        r.timings.output_head_ms = (time.monotonic() - to) * 1e3
        if out.host_buf_offset == 0 or out.data_bytes < 4 * R:
            raise BridgeError(
                f"batched-verify head readback missing: off="
                f"{out.host_buf_offset} bytes={out.data_bytes}")
        r.argmax = self.read_head_readback_ids(out.host_buf_offset, R)
        for b, t in enumerate(r.argmax):
            if t >= self.vocab_size:
                raise BridgeError(f"verify argmax row {b} out of vocab: {t}")

        r.timings.total_ms = (time.monotonic() - t_start) * 1e3
        return r

    def prefill_chunk_fetch_and_run(self, toks: list[int], seq_id: int,
                                    pos0: int,
                                    lrus: list[GpuLru] | None
                                    ) -> int:
        """One prompt prefill chunk (no output head). Returns Σ lookups
        (excluded from decode hit stats by the callers, C++ parity).

        Timeout: at least the historical 120 s, raised to the boot's
        decode deadline when larger (deepseek_v4 sets 200 s — a 512-row
        chunk's expert UNION approaches the full per-layer set streamed
        from the GGUF page cache; GLM byte-identical at the 5 s default).
        """
        lookups = 0
        timeout_us = max(120_000_000, self.decode_timeout_us)
        self._embed(toks)
        n = len(toks)
        self.write_batch_descriptors(seq_id, pos0, n)
        if self.use_far and lrus is None and self.far_burst:
            return lookups + self._far_sweep_burst(
                n, is_prefill=1, chunk_start=pos0, chunk_len=n,
                timeout_us=timeout_us)
        for layer in range(self.num_layers):
            is_moe = layer >= self.first_moe_layer
            if self.use_far and lrus is None:
                lookups += self._far_layer(
                    layer, n, is_prefill=1, chunk_start=pos0, chunk_len=n,
                    timeout_us=timeout_us)
                continue
            self._run_attention(layer, n, is_prefill=1, chunk_start=pos0,
                                chunk_len=n, is_moe=is_moe)
            if is_moe:
                lookups += self._routed_moe(layer, n, lrus,
                                            timeout_us=timeout_us)
            else:
                self._run_dense_moe(layer, n)
        return lookups

    def prefill_superchunk_fetch_and_run(self, toks: list[int], seq_id: int,
                                         pos0: int, sub: int) -> int:
        """SC (superchunk port): one SUPERCHUNK over len(toks) prompt tokens
        at positions [pos0, pos0+len), processed LAYER-WISE (the GLM
        TD-PREFILL-MOE-BIG driver shape): embedding per sub-chunk at
        row_offset, then per layer K attention sub-launches (superchunk
        flag + row_offset; fused-gate topk stored at row offsets, per
        sub-chunk exports unioned here in first-occurrence order) and ONE
        E_CMD_FETCH_AND_RUN_MOE_BIG over ALL rows — one fetch per unique
        expert per layer per superchunk. Returns Σ deduped entry counts.

        Caller contract: len(toks) <= EngineInfo.moe_batch_capacity, and
        sub <= 512 (MAX_BATCH_DESCRIPTORS / MAX_SIDEBAND_TOKEN_IDS).
        Entry assignment follows route_arm: "act" (V4) = e % moe_gpus, no
        evict map; "reef" (GLM serving-prefill lever, 2026-08-23) = the
        daemon ReefOrch service via E_CMD_REEF_ROUTE per layer union —
        REEF-consistent placement + the 13c-2.0 victim map, so prefill
        expert traffic lands where the decode-time placement lives
        (arbitrary e%tp spread would churn the REEF-placed stable zone).
        """
        n = len(toks)
        sub = max(1, min(sub, MAX_BATCH_DESCRIPTORS))
        timeout_us = max(120_000_000, self.decode_timeout_us)
        lookups = 0
        # 1. Embedding sub-chunks into hidden rows [off, off+len).
        for off in range(0, n, sub):
            self._embed(toks[off:off + sub], row_offset=off)
        # 2. Layer sweep: K attention sub-launches + ONE MOE_BIG per layer.
        for layer in range(self.num_layers):
            is_moe = layer >= self.first_moe_layer
            seen: set[int] = set()
            union: list[int] = []
            for off in range(0, n, sub):
                ln = min(sub, n - off)
                self.write_batch_descriptors(seq_id, pos0 + off, ln)
                self._run_attention(layer, ln, is_prefill=1,
                                    chunk_start=pos0 + off, chunk_len=ln,
                                    is_moe=is_moe, superchunk=1,
                                    row_offset=off)
                if is_moe:
                    for e in self.read_routing_union(layer, ln):
                        if e not in seen:
                            seen.add(e)
                            union.append(e)
            if not is_moe:
                self._run_dense_moe(layer, n)
                continue
            if not union:
                raise BridgeError(f"no routed experts exported L{layer} "
                                  "(superchunk)")
            if self.route_arm == "reef":
                count = self._write_moe_entries(
                    layer, union, [0] * len(union),
                    [(layer, 0xFFFF, 0)] * len(union))
                c = self._cmd(E_CMD_REEF_ROUTE)
                c.payload.reef_route.layer_idx = layer
                c.payload.reef_route.expert_count = count
                self.send(c)
                out = self.wait(CMP_COMPUTE_DONE,
                                ctx=f"reef route L{layer} (sc)")
                if out.status != 0:
                    raise BridgeError(
                        f"reef route L{layer} (sc) status {out.status}")
                self._fetch_and_run_moe_big(layer, n, count,
                                            timeout_us=timeout_us,
                                            have_evict_map=True)
            else:
                assigns = [e % self.moe_gpus for e in union]
                victims = [(layer, 0xFFFF, g) for g in assigns]
                count = self._write_moe_entries(layer, union, assigns,
                                                victims)
                self._fetch_and_run_moe_big(layer, n, count,
                                            timeout_us=timeout_us)
            lookups += count
        return lookups

    # ── DSpark draft step (DSP-5 command seam) ───────────────────────────

    def _dspark_cmd(self, seq_id: int, anchor_token: int, anchor_pos: int,
                    gamma: int) -> Command:
        c = self._cmd(D_CMD_RUN_DSPARK_STEP)
        p = c.payload.run_dspark_step
        p.seq_id = seq_id
        p.anchor_token_id = anchor_token
        p.anchor_pos = anchor_pos
        p.num_query = gamma
        p.step_idx = 0
        return c

    def _parse_dspark(self, out: Cmp, gamma: int, with_conf: bool
                      ) -> tuple[list[int], list[float]]:
        # Draft READBACK defects are draft-side only (the ids/confs never
        # reach a target feed unless the caller verifies them) — classified
        # DsparkDraftError so serving can fall back instead of failing.
        want = gamma * 4 * (2 if with_conf else 1)
        if out.host_buf_offset == 0 or out.data_bytes < want:
            raise DsparkDraftError(
                f"dspark step readback missing: off={out.host_buf_offset} "
                f"bytes={out.data_bytes} (want >= {want})")
        ids, confs = self.read_spec_readback_ids(out.host_buf_offset, gamma,
                                                 with_conf)
        for k, t in enumerate(ids):
            if t < 0 or t >= self.vocab_size:
                raise DsparkDraftError(
                    f"dspark draft id out of vocab: d_{k}={t}")
        if with_conf:
            for k, cf in enumerate(confs):
                if not math.isfinite(cf) or cf <= 0.0 or cf >= 1.0:
                    raise DsparkDraftError(
                        f"dspark confidence c_{k} outside (0,1): {cf}")
        return ids, confs

    def _send_dspark(self, seq_id: int, anchor_token: int, anchor_pos: int,
                     gamma: int) -> int:
        """Issue D_CMD_RUN_DSPARK_STEP; returns its cmd_seq."""
        if self._v2:
            seq = self._next_seq()
            if not _fb.send_dspark(*self._cargs, seq, seq_id, anchor_token,
                                   anchor_pos, gamma):
                raise BridgeError("command ring full")
            return seq
        c = self._dspark_cmd(seq_id, anchor_token, anchor_pos, gamma)
        self.send(c)
        return int(c.cmd_seq)

    def dspark_draft_step(self, seq_id: int, anchor_token: int,
                          anchor_pos: int, gamma: int, with_conf: bool
                          ) -> tuple[list[int], list[float]]:
        self._send_dspark(seq_id, anchor_token, anchor_pos, gamma)
        out = self.wait(CMP_COMPUTE_DONE, ctx="dspark step")
        if out.status != 0:
            raise BridgeError(f"dspark step status {out.status}")
        return self._parse_dspark(out, gamma, with_conf)

    def dspark_send_async(self, seq_id: int, anchor_token: int,
                          anchor_pos: int, gamma: int) -> None:
        self._dspark_cmp = None
        self._dspark_err = None
        self._dspark_pending_seq = self._send_dspark(
            seq_id, anchor_token, anchor_pos, gamma)

    def dspark_collect_async(self, gamma: int, with_conf: bool,
                             timeout_s: float = 300.0
                             ) -> tuple[list[int], list[float]]:
        if self._dspark_err is not None:
            # wait() consumed the async draft's CMP_ERROR while servicing
            # a target command and stashed it — surface it HERE, at the
            # round's draft boundary.
            msg = self._dspark_err
            self._dspark_err = None
            raise DsparkDraftError(msg)
        if self._dspark_cmp is None:
            deadline = time.monotonic() + timeout_s
            while self._dspark_cmp is None:
                if time.monotonic() >= deadline:
                    self._dspark_pending_seq = 0
                    raise BridgeError("timeout collecting async dspark step")
                out = self._poll()
                if out is None:
                    os.sched_yield()
                    continue
                if (self.fire_forget_seqs
                        and out.cmd_seq in self.fire_forget_seqs):
                    self.fire_forget_seqs.discard(out.cmd_seq)
                    if out.cmp_type == CMP_ERROR:
                        print(f"  [bridge PF] prefetch error: {out.err_msg}",
                              flush=True)
                    continue
                if out.cmp_type == CMP_ERROR:
                    if out.cmd_seq == self._dspark_pending_seq:
                        # The draft step's own failure — draft-side class.
                        self._dspark_pending_seq = 0
                        raise DsparkDraftError(
                            f"CMP_ERROR (dspark collect): {out.err_msg}")
                    self._dspark_pending_seq = 0
                    raise BridgeError(
                        f"CMP_ERROR (dspark collect): {out.err_msg}")
                if out.cmp_type == CMP_CHECKPOINT:
                    continue
                if out.cmd_seq == self._dspark_pending_seq:
                    self._dspark_cmp = out
                    self._dspark_pending_seq = 0
                    break
                raise BridgeError(
                    f"unexpected completion (type 0x{out.cmp_type:x} seq "
                    f"{out.cmd_seq}) while collecting async dspark step")
        out = self._dspark_cmp
        self._dspark_cmp = None
        return self._parse_dspark(out, gamma, with_conf)

    def drain_pending_dspark(self, gamma: int) -> None:
        """Drain an uncollected async draft so its completion cannot leak
        into later ring waits (overlap-mode early break, C++ parity)."""
        if self._dspark_pending_seq and self._dspark_cmp is None:
            try:
                self.dspark_collect_async(gamma, False, timeout_s=30.0)
            except BridgeError as e:
                print(f"  [bridge] pending dspark drain: {e}", flush=True)
        self._dspark_cmp = None
        self._dspark_err = None                  # never leak across requests


# ── import-time layout self-check ───────────────────────────────────────────
# The offsets the (optional) Cython fast path packs are asserted against the
# ctypes protocol mirror here, once, so silent header drift fails loud.

def _layout_selfcheck() -> None:
    assert Command.payload.offset == 16
    assert ctypes.sizeof(Command) == CMD_SLOT_BYTES
    assert ctypes.sizeof(Completion) == CMP_SLOT_BYTES
    assert Completion.payload.offset == 16
    assert ctypes.sizeof(BatchDescriptorEntry) == 16
    assert ctypes.sizeof(ExpertPrefetchEntry) == 8
    assert ctypes.sizeof(ExpertEvictionEntry) == 8
    assert _PRODUCER_OFF == 0
    assert _CONSUMER_OFF == 64

    # v2 hot path packs Command payload fields at fixed absolute offsets
    # (see _fastbridge.pyx). Probe each through the ctypes mirror: set the
    # field, find the sentinel byte pattern at the expected offset.
    import struct

    def probe(setter, offset: int, fmt: str, value) -> None:
        c = Command()
        setter(c, value)
        raw = bytes(c)
        got = struct.unpack_from(fmt, raw, offset)[0]
        assert got == value, (
            f"v2 layout drift: expected {value!r} at +{offset} ({fmt}), "
            f"got {got!r} — regenerate/realign _fastbridge.pyx")

    P = lambda c: c.payload  # noqa: E731
    probe(lambda c, v: setattr(P(c).embedding_lookup, "num_tokens", v),
          16, "<I", 0xA1B2C3D4)
    probe(lambda c, v: setattr(P(c).embedding_lookup, "output_buf_id", v),
          20, "<I", 0xA1B2C3D4)
    probe(lambda c, v: setattr(P(c).embedding_lookup, "row_offset", v),
          24, "<I", 0xA1B2C3D4)   # SC superchunk staging row
    ra = [("layer_idx", 16, "<I", 0xA1B2C3D4),
          ("num_seqs", 20, "<I", 0xA1B2C3D4),
          ("is_prefill", 24, "<B", 0xA5),
          ("chunk_start", 28, "<I", 0xA1B2C3D4),
          ("chunk_len", 32, "<I", 0xA1B2C3D4),
          ("emit_gating", 36, "<B", 0xA5),
          ("store_gating", 37, "<B", 0xA5),
          ("superchunk", 38, "<B", 0xA5),      # SC sub-launch flag
          ("row_offset", 40, "<I", 0xA1B2C3D4)]
    for name, off, fmt, val in ra:
        probe(lambda c, v, n=name: setattr(P(c).run_attention, n, v),
              off, fmt, val)
    rm = [("layer_idx", 16, "<I", 0xA1B2C3D4), ("num_seqs", 20, "<I", 7)]
    for name, off, fmt, val in rm:
        probe(lambda c, v, n=name: setattr(P(c).run_moe, n, v),
              off, fmt, val)
    fm = [("layer_idx", 16, "<I", 0xA1B2C3D4),
          ("num_seqs", 20, "<I", 0xA1B2C3D4),
          ("expert_count", 24, "<I", 0xA1B2C3D4),
          ("timeout_us", 28, "<I", 0xA1B2C3D4),
          ("have_evict_map", 38, "<B", 0xA5)]
    for name, off, fmt, val in fm:
        probe(lambda c, v, n=name: setattr(P(c).fetch_and_run_moe, n, v),
              off, fmt, val)
    oh = [("num_tokens", 16, "<I", 0xA1B2C3D4),
          ("input_buf_id", 20, "<I", 0xA1B2C3D4),
          ("output_buf_id", 24, "<I", 0xA1B2C3D4),
          ("readback_to_host", 28, "<B", 0xA5),
          ("compute_confidence", 29, "<B", 0xA5),
          ("readback_logits", 32, "<B", 0xA5)]
    for name, off, fmt, val in oh:
        probe(lambda c, v, n=name: setattr(P(c).output_head, n, v),
              off, fmt, val)
    st = [("num_tokens", 16, "<I", 0xA1B2C3D4),
          ("logits_buf_id", 20, "<I", 0xA1B2C3D4),
          ("vocab_size", 24, "<I", 0xA1B2C3D4),
          ("random_seed", 40, "<Q", 0x1122334455667788)]
    for name, off, fmt, val in st:
        probe(lambda c, v, n=name: setattr(P(c).sample_tokens, n, v),
              off, fmt, val)
    probe(lambda c, v: setattr(P(c).sample_tokens, "top_p", v),
          36, "<f", 1.0)
    probe(lambda c, v: setattr(P(c).seq_create, "seq_id", v),
          16, "<Q", 0x1122334455667788)
    probe(lambda c, v: setattr(P(c).seq_create, "prompt_len", v),
          24, "<I", 0xA1B2C3D4)
    probe(lambda c, v: setattr(P(c).seq_free, "seq_id", v),
          16, "<Q", 0x1122334455667788)
    probe(lambda c, v: setattr(P(c).seq_fork, "src_seq_id", v),
          16, "<Q", 0x1122334455667788)
    probe(lambda c, v: setattr(P(c).seq_fork, "dst_seq_id", v),
          24, "<Q", 0x1122334455667788)
    ds = [("seq_id", 16, "<Q", 0x1122334455667788),
          ("anchor_token_id", 24, "<I", 0xA1B2C3D4),
          ("anchor_pos", 28, "<I", 0xA1B2C3D4),
          ("num_query", 32, "<B", 0xA5)]
    for name, off, fmt, val in ds:
        probe(lambda c, v, n=name: setattr(P(c).run_dspark_step, n, v),
              off, fmt, val)
    probe(lambda c, v: setattr(P(c).prefetch_batch, "count", v),
          16, "<I", 0xA1B2C3D4)
    probe(lambda c, v: setattr(P(c).prefetch_batch, "priority", v),
          20, "<f", -1.0)
    for name, off in (("layer_idx", 16), ("expert_count", 20)):
        probe(lambda c, v, n=name: setattr(P(c).reef_route, n, v),
              off, "<I", 0xA1B2C3D4)
    for name, off in (("layer_idx", 16), ("num_seqs", 20),
                      ("chunk_start", 24), ("chunk_len", 28),
                      ("timeout_us", 32)):
        probe(lambda c, v, n=name: setattr(P(c).far_forward_layer, n, v),
              off, "<I", 0xA1B2C3D4)
    for name, off in (("is_prefill", 36), ("route_mode", 37)):
        probe(lambda c, v, n=name: setattr(P(c).far_forward_layer, n, v),
              off, "<B", 0xA5)


_layout_selfcheck()
