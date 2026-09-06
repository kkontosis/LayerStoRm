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
    CMD_SEQ_FORK_FROZEN,
    CMD_SEQ_FREE,
    CMD_SEQ_HIBERNATE,
    CMP_COMPUTE_DONE,
    CMP_ERROR,
    CMP_SEQ_OP_DONE,
    Command,
    Completion,
    D_B_CMD_PREFETCH_BATCH,
    D_B_CMD_RUN_ATTENTION,
    D_B_CMD_RUN_MOE,
    D_CMD_KDA_CKPT,
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
        self.frozen_forks = 0
        self.last_fork_prefix = -1   # R4a: prefix_len of the last fork
        self.fork_prefixes: list[int] = []   # every fork's prefix_len
        self.hibernates: list[int] = []
        # P-29 step 24 (LS_KDA_PREFIX_CKPT): per-seq host checkpoint
        # positions (mirrors SequenceState::kda_ckpts), the scripted
        # per-checkpoint byte size, capture log, and the KDA-arch switch:
        # when kda_arch is True a TRUNCATING fork is admitted ONLY at a
        # recorded checkpoint position (the engine's reworked gate);
        # False keeps every pre-existing test byte-identical (truncation
        # freely admitted, the arch-neutral fake).
        self.kda_arch = False
        self.kda_ckpts: dict[int, set[int]] = {}
        self.kda_ckpt_bytes_each = 1 << 20
        self.kda_ckpt_captures: list[tuple[int, int]] = []
        # TD-PREFIX-TIDY-COLD-SPILL: scripted spill behavior — list of
        # (status, spilled_pages) answers consumed per spill=1 hibernate
        # (empty = engine no-tiering arm: status 0, 0 pages).
        self.spills: list[tuple[int, int]] = []
        self.spill_answers: list[tuple[int, int]] = []
        self.seq_frees = 0
        # Page-pool admission emulation (evict-at-admission tests): when
        # set, SEQ_CREATE / SEQ_FORK are rejected with the engine's
        # "pool exhausted" CMP_ERROR while len(known_seqs) >= seq_capacity
        # (a live holder pins its "pages" until SEQ_FREE'd — the
        # regression-hunt 2026-08-23 finding (b) shape).
        self.seq_capacity: int | None = None
        self.seq_admission_rejects = 0
        # Scripted kIndexerK exhaustion (TD-INDEXER-POOL-EVICT): every
        # RUN_ATTENTION is declined with the engine's RETRYABLE
        # pool-exhaustion CMP_ERROR until `seq_frees` reaches this mark —
        # i.e. until the orchestrator has freed a prefix holder, whose
        # indexer-K pages die with its sequence. Models the real seam: the
        # engine fail-closes the step rather than downgrading a KV-demoted
        # sequence to dense. None = never fail.
        self.indexer_exhaust_until_frees: int | None = None
        self.indexer_exhaust_rejects = 0
        # TD-INDEXER-NO-DENSE-FALLBACK (Route 1) emulation: record the
        # reservation each SEQ_CREATE/SEQ_FORK carried, grant
        # min(reserve, max_seq_tokens) back through seq_op.reserved_tokens
        # (aliases compute.data_bytes on the wire), and optionally script
        # an ADMISSION-TIME reservation exhaustion (category 29) until
        # `seq_frees` reaches the mark — the create-side evict-retry shape.
        self.max_seq_tokens = 25600
        self.last_create_reserve = -1
        self.last_fork_reserve = -1
        self.reserve_exhaust_until_frees: int | None = None
        self.reserve_exhaust_rejects = 0
        # Scripted indexer_dense witness bytes: the next N RUN_ATTENTION
        # completions carry compute.indexer_dense=1 (a dense kDead step).
        self.dense_flag_remaining = 0
        # Scripted V4 side-tier exhaustion (INV-PREFIX-CACHE-3 seam): same
        # gate as indexer_exhaust_until_frees, but the CMP_ERROR replays
        # the 2026-08-26 incident EXACTLY — the engine's 100-byte message
        # TRUNCATED by the 80-byte CMP field to "...pool exha" (NO
        # "exhausted" substring!) with error_category=kKvPoolExhausted(29).
        # Retryability must be recognized from the CATEGORY.
        self.v4_tier_exhaust_until_frees: int | None = None
        self.v4_tier_exhaust_rejects = 0
        # Scripted MID-SPEC-ROUND exhaustion (TD-SPEC-ROUND-POOL-EVICT):
        # replay the 2026-08-26 truncated CMP byte-for-byte, but WHILE a
        # dspark draft is in flight.  When set to the 0-based
        # RUN_DSPARK_STEP call index N, that draft's completion is HELD
        # (draft logically "in flight"); every attention-carrying command
        # arriving while it is held is declined with the TRUNCATED V4
        # side-tier CMP_ERROR (category kKvPoolExhausted=29 — retryability
        # must come from the CATEGORY, the 80-byte field ate "exhausted").
        # The held completion is released when a CMD_SEQ_FREE arrives —
        # written BEFORE the free's own CMP_SEQ_OP_DONE, replaying the
        # engine interleave where the draft completes while the
        # orchestrator's holder-eviction free is waiting (the bridge
        # wait() must stash it).  spec_round_exhaust_max_rejects bounds
        # the declines: after that many, the held completion is released
        # WITHOUT a SEQ_FREE (models the draft finishing on its own while
        # nothing is evictable).  spec_round_draft_error makes the held
        # completion the draft's own CMP_ERROR instead (draft-side
        # failure surfacing mid-eviction — must stash + surface as
        # DsparkDraftError at collect).
        self.spec_round_exhaust_at_call: int | None = None
        self.spec_round_exhaust_rejects = 0
        self.spec_round_exhaust_max_rejects: int | None = None
        self.spec_round_draft_error = False
        self.held_dspark_releases = 0
        self._held_dspark: bytes | None = None
        # Scripted draft-context invalidation (TD-DSPARK-CTX-CAP /
        # INV-SERVE-SPEC-FALLBACK): from dspark call index N on, every
        # D_CMD_RUN_DSPARK_STEP is declined with the engine's real
        # run_step CMP_ERROR (sticky, like the runtime — an invalidated
        # context never recovers within a request; positions only grow).
        # None = never fail; 0 = invalid from the first draft.
        self.dspark_fail_from: int | None = None
        self.embed_calls: list[int] = []     # n per EMBEDDING_LOOKUP
        # TD-MOE-PROGRESSIVE-DEGRADED-SILENT: while > 0, each MoE-carrying
        # completion (FETCH_AND_RUN_MOE / _BIG / FAR on a MoE layer) is
        # emitted with compute.moe_degraded=1 and this counts down —
        # scripting the engine's degraded progressive finalize.
        self.degrade_moe_remaining = 0
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

    def _build_done(self, cmd: Command, **compute) -> Completion:
        c = Completion()
        c.cmp_type = (CMP_SEQ_OP_DONE
                      if cmd.cmd_type in (CMD_SEQ_CREATE, CMD_SEQ_FREE,
                                          CMD_SEQ_FORK, CMD_SEQ_FORK_FROZEN,
                                          CMD_SEQ_HIBERNATE)
                      else CMP_COMPUTE_DONE)
        c.cmd_seq = cmd.cmd_seq
        c.gpu_idx = 0
        c.status = compute.pop("status", 0)  # envelope field, not payload
        c.payload.compute.cmd_type = cmd.cmd_type
        for k, v in compute.items():
            setattr(c.payload.compute, k, v)
        return c

    def _done(self, cmd: Command, **compute) -> None:
        self._write_cmp(self._build_done(cmd, **compute))

    def _degrade_flag(self) -> int:
        """Consume one scripted degraded-MoE completion, if any."""
        if self.degrade_moe_remaining > 0:
            self.degrade_moe_remaining -= 1
            return 1
        return 0

    def run(self) -> None:
        try:
            while not self.stop_flag:
                cmd = self._read_cmd()
                if cmd is None:
                    continue
                self._handle(cmd)
        except Exception as e:  # surfaced by the test after join
            self.errors.append(repr(e))

    def _error(self, cmd: Command, msg: str, category: int = 2) -> None:
        c = Completion()
        c.cmp_type = CMP_ERROR
        c.cmd_seq = cmd.cmd_seq
        c.gpu_idx = 0
        c.status = 1
        c.payload.error.error_category = category
        # Engine-faithful: the CMP message field is 80 bytes (NUL-
        # terminated) — long messages truncate exactly like write_error's.
        c.payload.error.message = msg.encode()[:79]
        self._write_cmp(c)

    # The 2026-08-26 regression message, byte-for-byte as delivered: the
    # engine's 100-byte string cut at 79 bytes by the CMP field.
    V4_TIER_EXHAUST_MSG = ("attention: V4 side-tier page provisioning "
                           "failed (kSwa/kHca/kIndexerK pool exhausted "
                           "— fail-closed)")
    ERR_CAT_KV_POOL_EXHAUSTED = 29

    def _v4_tier_exhausted(self, cmd: Command) -> bool:
        if (self.v4_tier_exhaust_until_frees is not None
                and self.seq_frees < self.v4_tier_exhaust_until_frees):
            self.v4_tier_exhaust_rejects += 1
            self._error(cmd, self.V4_TIER_EXHAUST_MSG,
                        category=self.ERR_CAT_KV_POOL_EXHAUSTED)
            return True
        return False

    def _admission_full(self) -> bool:
        return (self.seq_capacity is not None
                and len(self.known_seqs) >= self.seq_capacity)

    def _release_held_dspark(self) -> None:
        if self._held_dspark is not None:
            self._write_cmp(self._held_dspark)   # bytes(bytes) is identity
            self._held_dspark = None
            self.held_dspark_releases += 1

    def _spec_round_exhausted(self, cmd: Command) -> bool:
        """Decline an attention-carrying command while a draft completion
        is held — the byte-exact truncated V4 CMP (category 29)."""
        if self._held_dspark is None:
            return False
        self.spec_round_exhaust_rejects += 1
        self._error(cmd, self.V4_TIER_EXHAUST_MSG,
                    category=self.ERR_CAT_KV_POOL_EXHAUSTED)
        if (self.spec_round_exhaust_max_rejects is not None
                and (self.spec_round_exhaust_rejects
                     >= self.spec_round_exhaust_max_rejects)):
            self._release_held_dspark()      # draft finishes on its own
        return True

    def _handle(self, cmd: Command) -> None:
        t = cmd.cmd_type
        if t == CMD_SEQ_CREATE:
            if self._admission_full():
                self.seq_admission_rejects += 1
                self._error(cmd, "seq_create: kMain page pool exhausted "
                                 f"(scripted cap {self.seq_capacity})")
                return
            reserve = int(cmd.payload.seq_create.reserve_tokens)
            self.last_create_reserve = reserve
            if (reserve > 0
                    and self.reserve_exhaust_until_frees is not None
                    and self.seq_frees < self.reserve_exhaust_until_frees):
                self.reserve_exhaust_rejects += 1
                # Engine-faithful shape: category 29 (kKvPoolExhausted) +
                # "exhausted" early in the 80-byte message.
                self._error(cmd, "seq_create: exhausted indexer-K pool at "
                                 "admission reservation — retryable, evict "
                                 "a prefix holder", category=29)
                return
            self.known_seqs.add(int(cmd.payload.seq_create.seq_id))
            self._done(cmd, data_bytes=(min(reserve, self.max_seq_tokens)
                                        if reserve else 0))
        elif t == CMD_SEQ_FREE:
            # TD-SPEC-ROUND-POOL-EVICT interleave: the in-flight draft
            # completes while the orchestrator waits on this free — its
            # completion lands BEFORE the CMP_SEQ_OP_DONE, so the bridge
            # wait() must stash it for dspark_collect_async.
            self._release_held_dspark()
            self.known_seqs.discard(int(cmd.payload.seq_free.seq_id))
            self.kda_ckpts.pop(int(cmd.payload.seq_free.seq_id), None)
            self.seq_frees += 1
            self._done(cmd)
        elif t in (CMD_SEQ_FORK, CMD_SEQ_FORK_FROZEN):
            # Prefix-cache / spec-fork primitive: the fake is stateless per
            # seq (routing derives from the embedded rows), so a fork just
            # validates lifecycle and registers the child.  FROZEN forks
            # (R3 holder registration — engine-side CoW-free) are counted
            # separately so tests can assert the registration arm.
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
            prefix = int(cmd.payload.seq_fork.prefix_len)
            # P-29 step 24 gate mirror: on the KDA arch a truncating fork
            # needs a checkpoint at EXACTLY prefix_len (kSeqFork refusal
            # otherwise — the engine's reworked blanket gate).
            if (self.kda_arch and prefix > 0
                    and prefix not in self.kda_ckpts.get(src, set())):
                self._error(cmd, "seq_fork: prefix_len unsupported: no "
                                 "KDA checkpoint at prefix_len (lossy "
                                 "arch)", category=13)
                return
            self.known_seqs.add(dst)
            self.forks += 1
            # R4a: record the truncation prefix (0 = full fork) so tests
            # can assert what the wire carried.
            self.last_fork_prefix = prefix
            self.fork_prefixes.append(prefix)
            if t == CMD_SEQ_FORK_FROZEN and src in self.kda_ckpts:
                # Registration fork MOVES the parent's checkpoints to the
                # frozen holder (engine semantics, P-29 step 24).
                self.kda_ckpts[dst] = self.kda_ckpts.pop(src)
            reserve = int(cmd.payload.seq_fork.reserve_tokens)
            self.last_fork_reserve = reserve
            if t == CMD_SEQ_FORK_FROZEN:
                self.frozen_forks += 1
                reserve = 0          # frozen holders never reserve
            self._done(cmd, data_bytes=(min(reserve, self.max_seq_tokens)
                                        if reserve else 0))
        elif t == D_CMD_KDA_CKPT:
            # P-29 step 24: host-RAM KDA prefix checkpoint capture. The
            # fake validates lifecycle only (frontier/grid preconditions
            # are engine-tested in CommandDispatcherKdaState); duplicate
            # positions are idempotent with data_bytes=0, like the engine.
            sid = int(cmd.payload.kda_anchor.seq_id)
            cpos = int(cmd.payload.kda_anchor.pos)
            if sid not in self.known_seqs:
                self._error(cmd, "kda_ckpt: unknown sequence or no live "
                                 "KDA state", category=10)
                return
            have = self.kda_ckpts.setdefault(sid, set())
            fresh = cpos not in have
            have.add(cpos)
            self.kda_ckpt_captures.append((sid, cpos))
            self._done(cmd, data_bytes=(self.kda_ckpt_bytes_each
                                        if fresh else 0))
        elif t == CMD_SEQ_HIBERNATE:
            # R3 holder hibernation: no-op success on the fake (mirrors
            # the engine's no-tiering arm); lifecycle validated + counted.
            sid = int(cmd.payload.seq_hibernate.seq_id)
            klen = int(cmd.payload.seq_hibernate.kv_len)
            if sid not in self.known_seqs:
                self._error(cmd, "seq_hibernate: unknown seq_id",
                            category=12)
                return
            if klen == 0:
                self._error(cmd, "seq_hibernate: kv_len required",
                            category=12)
                return
            spill = int(getattr(cmd.payload.seq_hibernate, "spill", 0))
            if spill:
                st, pages = (self.spill_answers.pop(0)
                             if self.spill_answers else (0, 0))
                self.spills.append((sid, st, pages))
                self._done(cmd, status=st, data_bytes=pages)
            else:
                self.hibernates.append((sid, klen))
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
            if self._spec_round_exhausted(cmd):
                return
            if (self.indexer_exhaust_until_frees is not None
                    and self.seq_frees < self.indexer_exhaust_until_frees):
                self.indexer_exhaust_rejects += 1
                self._error(cmd, "indexer-K pool exhausted (demoted seq) "
                                 "— retryable, evict a prefix holder")
                return
            if self._v4_tier_exhausted(cmd):
                return
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
            idense = 0
            if self.dense_flag_remaining > 0:
                self.dense_flag_remaining -= 1
                idense = 1
            self._done(cmd, layer_idx=p.layer_idx, indexer_dense=idense)
        elif t == E_CMD_FETCH_AND_RUN_MOE:
            p = cmd.payload.fetch_and_run_moe
            if p.expert_count == 0:
                self.errors.append("FETCH_AND_RUN_MOE expert_count 0")
            self.fetch_moes += 1
            self._done(cmd, layer_idx=p.layer_idx,
                       moe_degraded=self._degrade_flag())
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
            self._done(cmd, layer_idx=p.layer_idx,
                       moe_degraded=self._degrade_flag())
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
            # The fused layer RUNS ATTENTION engine-side, so this is where a
            # kIndexerK exhaustion surfaces on the FAR path (same guard as
            # D_B_CMD_RUN_ATTENTION above).
            if self._spec_round_exhausted(cmd):
                return
            if (self.indexer_exhaust_until_frees is not None
                    and self.seq_frees < self.indexer_exhaust_until_frees):
                self.indexer_exhaust_rejects += 1
                self._error(cmd, "indexer-K pool exhausted (demoted seq) "
                                 "— retryable, evict a prefix holder")
                return
            if self._v4_tier_exhausted(cmd):
                return
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
            idense = 0
            if self.dense_flag_remaining > 0:
                self.dense_flag_remaining -= 1
                idense = 1
            self._done(cmd, layer_idx=p.layer_idx, data_bytes=count,
                       moe_degraded=(self._degrade_flag()
                                     if p.layer_idx >= FIRST_MOE else 0),
                       indexer_dense=idense)
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
            hold = (self.spec_round_exhaust_at_call is not None
                    and self.dspark_calls == self.spec_round_exhaust_at_call)
            if hold and self.spec_round_draft_error:
                # The held completion is the DRAFT's own failure — it must
                # be stashed by whatever wait it interrupts and surface as
                # DsparkDraftError at dspark_collect_async.
                self.dspark_calls += 1
                c = Completion()
                c.cmp_type = CMP_ERROR
                c.cmd_seq = cmd.cmd_seq
                c.gpu_idx = 0
                c.status = 1
                c.payload.error.error_category = 2
                c.payload.error.message = (
                    b"dspark run_step: draft context invalidated (scripted)")
                self._held_dspark = bytes(c)
                return
            if (self.dspark_fail_from is not None
                    and self.dspark_calls >= self.dspark_fail_from):
                self.dspark_calls += 1
                self._error(cmd, f"dspark run_step: no valid ingested "
                                 f"context for seq {int(p.seq_id)} "
                                 f"(tracked seq {int(p.seq_id)}, valid 0)")
                return
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
            c = self._build_done(
                cmd, host_buf_offset=SPEC_READBACK_OFF,
                data_bytes=4 * g * (2 if self.conf_enabled else 1),
                top1_prob=0.9, entropy=0.1)
            if hold:
                self._held_dspark = bytes(c)   # in flight until release
            else:
                self._write_cmp(c)
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


def test_truncated_fork_carries_prefix_len():
    # R4a: a truncating fork rides the generic ring path and the engine
    # sees the prefix; full forks (default and fastbridge hot path) carry
    # prefix_len == 0 — the legacy wire shape.
    bridge, daemon, _ = _make()
    try:
        bridge.create_sequence(1, 64)
        bridge.fork_sequence(1, 2, prefix_len=48)
        assert daemon.forks == 1 and daemon.frozen_forks == 0
        assert daemon.last_fork_prefix == 48
        bridge.fork_sequence(1, 3)
        assert daemon.forks == 2
        assert daemon.last_fork_prefix == 0
        bridge.fork_sequence(1, 4, frozen=True, prefix_len=32)
        assert daemon.frozen_forks == 1
        assert daemon.last_fork_prefix == 32
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


# ── TD-MOE-PROGRESSIVE-DEGRADED-SILENT: degraded finalize observability ────
# A degraded progressive-MoE finalize (Completion.compute.moe_degraded=1)
# must be COUNTED by the bridge, whatever wait() does with the completion —
# the orchestrator turns the monotonic counter into a per-request stat.


def _drive_degraded_counter(bridge, daemon) -> None:
    daemon.degrade_moe_remaining = 3
    st = SpecStats()
    out = run_speculative_loop(bridge, 1, 20, 4321, None, st,
                               gamma=5, vb="batched", overlap=False)
    assert out[:20] == chain(4321, 20)
    assert bridge.moe_degraded_layers == 3, (
        f"degraded finalizes not counted: {bridge.moe_degraded_layers}")
    # Healthy follow-up run: the monotonic counter must NOT advance.
    out2 = run_speculative_loop(bridge, 2, 20, 999, None, SpecStats(),
                                gamma=5, vb="batched", overlap=False)
    assert out2[:20] == chain(999, 20)
    assert bridge.moe_degraded_layers == 3, "healthy run advanced the counter"


def test_moe_degraded_counter():
    bridge, daemon, _ = _make(gamma=5)
    try:
        _drive_degraded_counter(bridge, daemon)
    finally:
        _finish(daemon)


@pytest.mark.skipif(not ring_bridge.fastbridge_active(),
                    reason="_fastbridge not built")
def test_moe_degraded_counter_pure_ctypes(monkeypatch):
    monkeypatch.setattr(ring_bridge, "_fb", None)
    bridge, daemon, _ = _make(gamma=5)
    try:
        _drive_degraded_counter(bridge, daemon)
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


# ── DsparkDraftError classification (INV-SERVE-SPEC-FALLBACK bridge seam):
# an async draft's CMP_ERROR must never abort a TARGET command's wait — it
# is stashed and surfaces at dspark_collect_async as DsparkDraftError, with
# every target completion consumed normally before and after. ──────────────


def _drive_dspark_error_stash(bridge, daemon) -> None:
    from bridge.ring_bridge import DsparkDraftError

    bridge.create_sequence(1, 1)
    try:
        # Overlap shape: draft async UNDER a plain step.  The daemon
        # declines the draft (cmd-ring FIFO puts its CMP_ERROR ahead of
        # the plain step's completions), so the plain step's waits consume
        # + STASH it — the step must return its result intact.
        seed = 4321
        bridge.dspark_send_async(1, seed, 1, 5)
        r = bridge.decode_step_fetch_and_run(seed, 1, 0, None)
        assert r.sampled_token == f(seed), "plain step lost under stash"
        with pytest.raises(DsparkDraftError, match="no valid ingested"):
            bridge.dspark_collect_async(5, False)
        assert bridge._dspark_err is None          # stash consumed
        assert bridge._dspark_pending_seq == 0
        # The bridge stays fully usable: next plain step is clean.
        r2 = bridge.decode_step_fetch_and_run(r.sampled_token, 1, 1, None)
        assert r2.sampled_token == f(r.sampled_token)
    finally:
        bridge.free_sequence(1)


def test_async_dspark_error_stashed_not_raised_in_target_wait():
    bridge, daemon, _ = _make(gamma=5)
    daemon.dspark_fail_from = 0
    try:
        _drive_dspark_error_stash(bridge, daemon)
    finally:
        _finish(daemon)


@pytest.mark.skipif(not ring_bridge.fastbridge_active(),
                    reason="_fastbridge not built")
def test_async_dspark_error_stash_pure_ctypes(monkeypatch):
    monkeypatch.setattr(ring_bridge, "_fb", None)
    bridge, daemon, _ = _make(gamma=5)
    daemon.dspark_fail_from = 0
    try:
        _drive_dspark_error_stash(bridge, daemon)
    finally:
        _finish(daemon)


def test_dspark_error_at_collect_classified():
    """Error consumed by collect's own poll (no intervening target wait)
    → same DsparkDraftError classification."""
    from bridge.ring_bridge import DsparkDraftError

    bridge, daemon, _ = _make(gamma=5)
    daemon.dspark_fail_from = 0
    try:
        bridge.create_sequence(1, 1)
        try:
            bridge.dspark_send_async(1, 4321, 1, 5)
            with pytest.raises(DsparkDraftError, match="no valid ingested"):
                bridge.dspark_collect_async(5, False)
            assert bridge._dspark_pending_seq == 0
        finally:
            bridge.free_sequence(1)
    finally:
        _finish(daemon)


def test_drain_pending_dspark_clears_error_stash():
    """A stashed draft error must never leak into a later request's
    collect: drain_pending_dspark (every spec arm's finally) clears it."""
    bridge, daemon, _ = _make(gamma=5)
    daemon.dspark_fail_from = 0
    try:
        bridge.create_sequence(1, 1)
        try:
            bridge.dspark_send_async(1, 4321, 1, 5)
            r = bridge.decode_step_fetch_and_run(4321, 1, 0, None)
            assert r.sampled_token == f(4321)
            assert bridge._dspark_err is not None  # stashed, uncollected
            bridge.drain_pending_dspark(5)
            assert bridge._dspark_err is None
        finally:
            bridge.free_sequence(1)
    finally:
        _finish(daemon)


def test_is_pool_exhaustion_matches_category_and_substring():
    """INV-PREFIX-CACHE-3 seam predicate: retryable pool exhaustion is
    recognized by CMP error CATEGORY first — the 80-byte CMP message field
    truncated the 2026-08-26 V4 side-tier message to "...pool exha",
    silently disarming the substring-only match — with the substring arm
    retained for seq_create/seq_fork exhaustion (their own categories,
    "exhausted" early in the message)."""
    from bridge.ring_bridge import (BridgeError, ERR_CAT_KV_POOL_EXHAUSTED,
                                    is_pool_exhaustion)
    truncated = FakeDaemon.V4_TIER_EXHAUST_MSG.encode()[:79].decode()
    assert "exhausted" not in truncated
    # Category carries it even when truncation ate the keyword.
    assert is_pool_exhaustion(
        BridgeError(truncated, category=ERR_CAT_KV_POOL_EXHAUSTED))
    # Substring arm: other categories with the keyword intact.
    assert is_pool_exhaustion(
        BridgeError("seq_fork: indexer-K pool exhausted during CoW split",
                    category=0))
    # Neither: not retryable.
    assert not is_pool_exhaustion(BridgeError(truncated, category=2))
    assert not is_pool_exhaustion(BridgeError("dispatch exception"))


def test_free_sequence_under_pending_async_draft_stashes_completion():
    """TD-SPEC-ROUND-POOL-EVICT bridge validation: a SEQ_FREE issued while
    an async dspark draft is IN FLIGHT (the mid-round holder eviction) must
    neither consume nor drop the draft's completion.  The scripted daemon
    releases the held draft completion INSIDE the free's wait — before the
    CMP_SEQ_OP_DONE — so wait() must stash it and dspark_collect_async must
    return it intact afterwards."""
    gamma = 4
    bridge, daemon, _ = _make(gamma=gamma)
    try:
        bridge.create_sequence(1, 8)
        bridge.create_sequence(2, 8)             # the "holder"
        daemon.spec_round_exhaust_at_call = 0    # hold the first draft
        bridge.dspark_send_async(1, 4321, 0, gamma)
        # The eviction shape: free the holder while the draft is pending.
        bridge.free_sequence(2)
        assert daemon.held_dspark_releases == 1
        assert daemon.seq_frees == 1
        ids, _ = bridge.dspark_collect_async(gamma, False)
        # FakeDaemon draft for call 0: wrong_at=0, chain elsewhere.
        t = f(4321)
        want = [(t + 1) % VOCAB]
        for _k in range(gamma - 1):
            t = f(t)
            want.append(t)
        assert ids == want, "stashed draft completion corrupted"
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_free_sequence_under_pending_draft_stashes_draft_error():
    """Same interleave, draft-side FAILURE arm: the draft's CMP_ERROR
    arriving inside the free's wait is stashed (never raised at the free,
    never lost) and surfaces as DsparkDraftError at the round's collect —
    the spec->plain fallback trigger (INV-SERVE-SPEC-FALLBACK)."""
    from bridge.ring_bridge import DsparkDraftError
    gamma = 4
    bridge, daemon, _ = _make(gamma=gamma)
    try:
        bridge.create_sequence(1, 8)
        bridge.create_sequence(2, 8)
        daemon.spec_round_exhaust_at_call = 0
        daemon.spec_round_draft_error = True
        bridge.dspark_send_async(1, 4321, 0, gamma)
        bridge.free_sequence(2)                  # must NOT raise
        assert daemon.held_dspark_releases == 1
        try:
            bridge.dspark_collect_async(gamma, False)
            raise AssertionError("stashed draft error was not surfaced")
        except DsparkDraftError as e:
            assert "draft context invalidated" in str(e)
        assert not daemon.errors, daemon.errors
    finally:
        _finish(daemon)


def test_mtp_project_payload_layout():
    """P-29 step 11: the Python ctypes mirror of ipc::Command.mtp_project must
    match the C++ 12-byte layout exactly — prev_src rides the former
    _pad[0] byte, so a drift here silently feeds prev_src=0 (attn_buf
    trunk hidden) instead of the head-normed hidden the probe measures
    against. Negative control: the struct must NOT have grown."""
    import ctypes
    from orchestrator.shm_protocol import MtpProjectPayload
    assert ctypes.sizeof(MtpProjectPayload) == 12
    offs = {f[0]: getattr(MtpProjectPayload, f[0]).offset
            for f in MtpProjectPayload._fields_}
    assert offs["mtp_layer_idx"] == 0
    assert offs["input_token_id"] == 4
    assert offs["step_idx"] == 8
    assert offs["hidden_row"] == 9
    assert offs["prev_src"] == 10
