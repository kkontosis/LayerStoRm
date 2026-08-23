"""Lock-free SPSC ring reader/writer and seqlock snapshot reader.

Operates on raw memory via ctypes. On x86-64, aligned 64-bit loads/stores are
naturally atomic; CPython's ctypes reads act as volatile (not optimized away).
Fence methods are no-ops under TSO but documented as upgrade points for nogil.
"""

from __future__ import annotations

import array
import ctypes
import dataclasses
import os

from orchestrator.shm_protocol import (
    CMP_SLOT_BYTES,
    CMD_SLOT_BYTES,
    RingHeader,
    StateSnapshot,
)

_RING_HEADER_SIZE = ctypes.sizeof(RingHeader)
_SNAPSHOT_SIZE = ctypes.sizeof(StateSnapshot)


def _release_fence() -> None:
    """Compiler fence (release). No-op on x86-64 TSO."""


def _acquire_fence() -> None:
    """Compiler fence (acquire). No-op on x86-64 TSO."""


class SpscRingWriter:
    """Producer side of an SPSC ring. Python writes commands to the daemon."""

    __slots__ = ("_header_addr", "_slots_base", "_slot_count", "_slot_size", "_mask")

    def __init__(self, base_addr: int, slot_count: int, slot_size: int = CMD_SLOT_BYTES) -> None:
        self._header_addr = base_addr
        self._slots_base = base_addr + _RING_HEADER_SIZE
        self._slot_count = slot_count
        self._slot_size = slot_size
        self._mask = slot_count - 1

    def write(self, data: bytes) -> bool:
        """Write one slot. Returns False if ring is full."""
        assert len(data) == self._slot_size
        header = RingHeader.from_address(self._header_addr)
        prod = header.producer_seq
        # Volatile read of consumer_seq (cross-thread)
        cons = ctypes.c_uint64.from_address(
            self._header_addr + RingHeader.consumer_seq.offset
        ).value
        if prod - cons >= self._slot_count:
            return False

        idx = prod & self._mask
        dest = self._slots_base + idx * self._slot_size
        ctypes.memmove(dest, data, self._slot_size)

        _release_fence()
        header.producer_seq = prod + 1
        return True

    def write_struct(self, cmd: ctypes.Structure) -> bool:
        """Write a ctypes Structure as one slot."""
        return self.write(bytes(cmd))

    def is_full(self) -> bool:
        header = RingHeader.from_address(self._header_addr)
        cons = ctypes.c_uint64.from_address(
            self._header_addr + RingHeader.consumer_seq.offset
        ).value
        return header.producer_seq - cons >= self._slot_count


class SpscRingReader:
    """Consumer side of an SPSC ring. Python reads completions from the daemon."""

    __slots__ = ("_header_addr", "_slots_base", "_slot_count", "_slot_size", "_mask")

    def __init__(self, base_addr: int, slot_count: int, slot_size: int = CMP_SLOT_BYTES) -> None:
        self._header_addr = base_addr
        self._slots_base = base_addr + _RING_HEADER_SIZE
        self._slot_count = slot_count
        self._slot_size = slot_size
        self._mask = slot_count - 1

    def read(self) -> bytes | None:
        """Read one slot. Returns None if ring is empty."""
        header = RingHeader.from_address(self._header_addr)
        cons = header.consumer_seq
        # Volatile read of producer_seq (cross-thread)
        prod = ctypes.c_uint64.from_address(
            self._header_addr + RingHeader.producer_seq.offset
        ).value
        _acquire_fence()
        if cons >= prod:
            return None

        idx = cons & self._mask
        src = self._slots_base + idx * self._slot_size
        buf = (ctypes.c_uint8 * self._slot_size).from_address(src)
        result = bytes(buf)

        _release_fence()
        header.consumer_seq = cons + 1
        return result

    def drain(self, max_count: int = 0xFFFFFFFF) -> list[bytes]:
        """Batch-read up to max_count slots."""
        header = RingHeader.from_address(self._header_addr)
        cons = header.consumer_seq
        prod = ctypes.c_uint64.from_address(
            self._header_addr + RingHeader.producer_seq.offset
        ).value
        _acquire_fence()

        avail = min(max_count, prod - cons)
        results: list[bytes] = []
        for i in range(avail):
            idx = (cons + i) & self._mask
            src = self._slots_base + idx * self._slot_size
            buf = (ctypes.c_uint8 * self._slot_size).from_address(src)
            results.append(bytes(buf))

        if avail > 0:
            _release_fence()
            header.consumer_seq = cons + avail
        return results

    def is_empty(self) -> bool:
        header = RingHeader.from_address(self._header_addr)
        prod = ctypes.c_uint64.from_address(
            self._header_addr + RingHeader.producer_seq.offset
        ).value
        return header.consumer_seq >= prod


@dataclasses.dataclass
class RetryPolicy:
    """Configurable retry params for transactional reads.

    Two yield-only stages (no sleep — avoids kernel timer overhead):
      Stage 0: stage_size spins, os.sched_yield() every stage0_yield_every iters
      Stage 1: remaining iters, os.sched_yield() per iter (~550ns each)

    Default max_retries=200M: ~3min practical, ~2min theoretical.
    Configurable via orchestrator.ipc_transaction in config.
    """
    max_retries: int = 200_000_000
    stage_size: int = 4096
    stage0_yield_every: int = 64


# Module-level default, overridable from config at engine init.
DEFAULT_RETRY_POLICY = RetryPolicy()


def set_default_retry_policy(max_retries: int, stage_size: int,
                             stage0_yield_every: int = 64) -> None:
    """Set module-level default retry policy from config."""
    global DEFAULT_RETRY_POLICY
    DEFAULT_RETRY_POLICY = RetryPolicy(max_retries=max_retries,
                                        stage_size=stage_size,
                                        stage0_yield_every=stage0_yield_every)


def _staged_backoff(iteration: int, policy: RetryPolicy) -> None:
    """Apply staged backoff based on current iteration count.

    Two stages only — no sleep, pure yield:
      Stage 0: yield every stage0_yield_every iterations
      Stage 1: yield every iteration
    """
    if iteration < policy.stage_size:
        # Stage 0: mostly tight spin, yield periodically
        if (iteration + 1) % policy.stage0_yield_every == 0:
            os.sched_yield()
    else:
        # Stage 1: yield every iteration (~550ns each)
        os.sched_yield()


class TransactionConflict(Exception):
    """Raised when a snapshot read transaction detects writer activity."""
    pass


class SnapshotTransaction:
    """Context manager for targeted snapshot reads with seqlock validation.

    Usage::

        with SnapshotTransaction(snap_addr) as tx:
            val = ctypes.c_uint32.from_address(snap_addr + offset).value
        # tx.__exit__ validates seqlock — raises TransactionConflict if writer was active

    For single-field reads with built-in retry, use the convenience methods
    (read_u8, read_u32, etc.) or the read_with_retry() helper.

    See spec/IPC_TX.md for the full design.
    """

    def __init__(self, snapshot_addr: int,
                 retry_policy: RetryPolicy | None = None) -> None:
        self._addr = snapshot_addr
        self._seq: int = 0
        self._policy = retry_policy or DEFAULT_RETRY_POLICY

    def begin(self) -> None:
        """Read seqlock.  Raises TransactionConflict if writer is active (odd)."""
        seq = ctypes.c_uint64.from_address(self._addr).value
        if seq & 1:
            raise TransactionConflict(f"seqlock odd ({seq})")
        _acquire_fence()
        self._seq = seq

    def end(self) -> None:
        """Validate seqlock unchanged.  Raises TransactionConflict on mismatch."""
        _acquire_fence()
        seq2 = ctypes.c_uint64.from_address(self._addr).value
        if seq2 != self._seq:
            raise TransactionConflict(
                f"seqlock changed ({self._seq} -> {seq2})")

    def __enter__(self) -> "SnapshotTransaction":
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.end()
        return False

    # ── Convenience: single-field reads with built-in transaction + retry ──

    def read_u8(self, offset: int) -> int:
        return self._read_single(offset, ctypes.c_uint8)

    def read_u32(self, offset: int) -> int:
        return self._read_single(offset, ctypes.c_uint32)

    def read_u64(self, offset: int) -> int:
        return self._read_single(offset, ctypes.c_uint64)

    def read_f32(self, offset: int) -> float:
        return self._read_single(offset, ctypes.c_float)

    def read_f64(self, offset: int) -> float:
        return self._read_single(offset, ctypes.c_double)

    def read_f32_array(self, offset: int, count: int) -> list[float]:
        """Read a contiguous float32 array with transaction + retry.

        The in-window work is ONE raw memcpy (ctypes.string_at, ~µs even
        for 128 KB); the Python float list is built AFTER seq2 validates.
        This is load-bearing, not an optimization: building the list
        element-by-element inside the window (the old ``list(arr)``) took
        longer than the daemon's publish interval once TD-IPC-MOE-LAYER-CAP
        grew the per-expert arrays to 128×256 entries — the seqlock could
        then NEVER validate and Phase-4 planning spun forever against a
        live daemon (INV-IPC-PUBLISH-THROTTLE's sibling failure mode on
        the READER side: keep the validation window ≪ publish interval).
        """
        addr = self._addr + offset
        nbytes = count * 4
        p = self._policy
        for i in range(p.max_retries):
            seq1 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 & 1:
                _staged_backoff(i, p)
                continue
            _acquire_fence()
            raw = ctypes.string_at(addr, nbytes)  # single memcpy
            _acquire_fence()
            seq2 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 == seq2:
                # Torn-read-safe: `raw` is a private copy, convert at leisure.
                return array.array("f", raw).tolist()
            _staged_backoff(i, p)
        raise TimeoutError(
            f"SnapshotTransaction.read_f32_array: exhausted {p.max_retries} retries")

    def read_f32_np(self, offset: int, count: int):
        """Read a contiguous float32 array as a numpy view over a PRIVATE copy.

        Same seqlock discipline as read_f32_array (one raw memcpy in the
        validation window, conversion after seq2 validates) but returns
        ``np.frombuffer`` over the copied bytes instead of a Python float
        list — 32k-entry stat arrays cost ~µs instead of ~ms (the per-cycle
        Phase-4 planning hot path, TD-ORCH-PLAN-CRAWL). The returned array
        is read-only (frombuffer over bytes) and owns no shared state.
        """
        import numpy as np
        addr = self._addr + offset
        nbytes = count * 4
        p = self._policy
        for i in range(p.max_retries):
            seq1 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 & 1:
                _staged_backoff(i, p)
                continue
            _acquire_fence()
            raw = ctypes.string_at(addr, nbytes)  # single memcpy
            _acquire_fence()
            seq2 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 == seq2:
                return np.frombuffer(raw, dtype=np.float32)
            _staged_backoff(i, p)
        raise TimeoutError(
            f"SnapshotTransaction.read_f32_np: exhausted {p.max_retries} retries")

    def _read_single(self, offset: int, ctype):
        addr = self._addr + offset
        p = self._policy
        for i in range(p.max_retries):
            seq1 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 & 1:
                _staged_backoff(i, p)
                continue
            _acquire_fence()
            val = ctype.from_address(addr).value
            _acquire_fence()
            seq2 = ctypes.c_uint64.from_address(self._addr).value
            if seq1 == seq2:
                return val
            _staged_backoff(i, p)
        raise TimeoutError(
            f"SnapshotTransaction._read_single: exhausted {p.max_retries} retries")


def read_with_retry(snapshot_addr: int, fn,
                    retry_policy: RetryPolicy | None = None):
    """Execute fn(tx) under a SnapshotTransaction, retrying on conflict.

    Uses staged exponential backoff per RetryPolicy. See spec/IPC_TX.md.

    Args:
        snapshot_addr: base address of the StateSnapshot
        fn: callable taking a SnapshotTransaction, returning a value
        retry_policy: override default retry params (None = use module default)
    """
    p = retry_policy or DEFAULT_RETRY_POLICY
    tx = SnapshotTransaction(snapshot_addr, retry_policy=p)
    for i in range(p.max_retries):
        try:
            with tx:
                return fn(tx)
        except TransactionConflict:
            _staged_backoff(i, p)
            continue
    raise TimeoutError(
        f"read_with_retry: exhausted {p.max_retries} retries")


def read_snapshot(snapshot_addr: int) -> StateSnapshot:
    """Read StateSnapshot with seqlock protection. Retries on torn reads.

    Backwards-compatible bulk-copy reader. Prefer targeted reads via
    SnapshotTransaction for performance-critical paths.
    """
    def _bulk_copy(tx: SnapshotTransaction) -> StateSnapshot:
        snap_copy = StateSnapshot()
        ctypes.memmove(ctypes.addressof(snap_copy), snapshot_addr, _SNAPSHOT_SIZE)
        return snap_copy
    return read_with_retry(snapshot_addr, _bulk_copy)
