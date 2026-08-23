"""High-level domain-specific reads from the IPC StateSnapshot.

All read methods are transaction-safe (retry on seqlock conflict) using the
SnapshotTransaction from spsc_ring.py per spec/IPC_TX.md.

Scalar reads use SnapshotTransaction convenience methods.
Multi-field reads use read_with_retry for atomic consistency.
Bulk array reads use SnapshotTransaction.read_f32_array.
"""

from __future__ import annotations

import ctypes

from orchestrator.shm_protocol import (
    GpuSnapshot,
    MAX_EXPERTS,
    MAX_GPUS,
    MAX_MOE_LAYERS,
    MAX_NUMA,
    MAX_TRACKED_REQUESTS,
    RequestAcceptance,
    StateSnapshot,
)
from orchestrator.spsc_ring import (
    RetryPolicy,
    SnapshotTransaction,
    read_with_retry,
)

# ── Precomputed offsets from ctypes field descriptors ─────────────────────────

_OFF_DAEMON_CYCLE    = StateSnapshot.daemon_cycle_count.offset
_OFF_TIMESTAMP_NS    = StateSnapshot.timestamp_ns.offset
_OFF_GPUS            = StateSnapshot.gpus.offset
_OFF_NUM_GPUS        = StateSnapshot.num_gpus.offset
_OFF_RESIDENCY       = StateSnapshot.residency_bitmap.offset
_OFF_GPU_TIER        = StateSnapshot.expert_gpu_tier.offset
_OFF_INTEREST        = StateSnapshot.expert_interest_count.offset
_OFF_HOST_BITMAP     = StateSnapshot.host_resident_bitmap.offset
_OFF_HOST_TIER       = StateSnapshot.host_tier.offset
_OFF_FREQ            = StateSnapshot.expert_frequency.offset
_OFF_RECENCY         = StateSnapshot.expert_recency.offset
_OFF_ROUTING_WT      = StateSnapshot.expert_routing_weight.offset
_OFF_AUTOCORR        = StateSnapshot.expert_temporal_autocorr.offset
_OFF_GLOBAL_ACC      = StateSnapshot.global_acceptance_rate.offset
_OFF_WINDOWED_ACC    = StateSnapshot.windowed_acceptance_rate.offset
_OFF_LAYERSKIP_ACC   = StateSnapshot.layer_skip_acceptance_rate.offset
_OFF_TOTAL_VERIF     = StateSnapshot.total_verifications.offset
_OFF_TOTAL_ACCEPTED  = StateSnapshot.total_accepted_tokens.offset
_OFF_TOTAL_ATTEMPTED = StateSnapshot.total_attempted_tokens.offset
_OFF_SHIFT           = StateSnapshot.shift_detected.offset
_OFF_INFLIGHT        = StateSnapshot.total_inflight_transfers.offset
_OFF_PER_REQ         = StateSnapshot.per_request_acceptance.offset
_OFF_NUM_TRACKED     = StateSnapshot.num_tracked_requests.offset
_OFF_HOST_NUMA       = StateSnapshot.expert_host_numa.offset
_OFF_HOST_NUMA_TIER  = StateSnapshot.host_numa_tier.offset
_OFF_LAST_CHANGE_NS  = StateSnapshot.expert_last_change_ns.offset

_GPU_SNAP_SIZE       = ctypes.sizeof(GpuSnapshot)
_REQ_ACCEPT_SIZE     = ctypes.sizeof(RequestAcceptance)


class StateReader:
    """Domain-specific reads from the IPC StateSnapshot via seqlock.

    Initialized with the snapshot base address (from ``EngineInfo``).
    All methods retry automatically on seqlock conflict.

    Usage::

        snap_addr = engine_info.ipc_base + engine_info.state_offset
        reader = StateReader(snap_addr)
        cycle = reader.cycle_count()
        freq = reader.expert_frequency(layer=0, expert=42)
    """

    __slots__ = ("_addr", "_policy")

    def __init__(self, snapshot_addr: int,
                 retry_policy: RetryPolicy | None = None) -> None:
        self._addr = snapshot_addr
        self._policy = retry_policy

    def _tx(self) -> SnapshotTransaction:
        return SnapshotTransaction(self._addr, retry_policy=self._policy)

    # ── Timestamps ────────────────────────────────────────────────────────

    def cycle_count(self) -> int:
        """Daemon cycle counter (monotonic)."""
        return self._tx().read_u64(_OFF_DAEMON_CYCLE)

    def timestamp_ns(self) -> int:
        """Nanosecond timestamp of last daemon publish."""
        return self._tx().read_u64(_OFF_TIMESTAMP_NS)

    # ── GPU metrics ───────────────────────────────────────────────────────

    def num_gpus(self) -> int:
        """Number of valid GPU entries in the snapshot."""
        return self._tx().read_u32(_OFF_NUM_GPUS)

    def gpu_snapshot(self, gpu_idx: int) -> GpuSnapshot:
        """Full GpuSnapshot struct for one GPU (56 bytes, atomic read)."""
        offset = _OFF_GPUS + gpu_idx * _GPU_SNAP_SIZE
        addr = self._addr
        buf = bytearray(_GPU_SNAP_SIZE)
        c_buf = (ctypes.c_char * _GPU_SNAP_SIZE).from_buffer(buf)

        def _copy(_tx: SnapshotTransaction) -> None:
            ctypes.memmove(c_buf, addr + offset, _GPU_SNAP_SIZE)

        read_with_retry(addr, _copy, self._policy)
        return GpuSnapshot.from_buffer_copy(buf)

    def vram_usage(self, gpu_idx: int) -> tuple[int, int]:
        """(used_bytes, total_bytes) for one GPU."""
        base = _OFF_GPUS + gpu_idx * _GPU_SNAP_SIZE
        off_used = base + GpuSnapshot.vram_used_bytes.offset
        off_total = base + GpuSnapshot.vram_total_bytes.offset
        addr = self._addr

        def _read(_tx: SnapshotTransaction) -> tuple[int, int]:
            u = ctypes.c_uint64.from_address(addr + off_used).value
            t = ctypes.c_uint64.from_address(addr + off_total).value
            return (u, t)

        return read_with_retry(addr, _read, self._policy)

    def cache_fill(self, gpu_idx: int) -> tuple[int, int, int, int]:
        """(stable_used, stable_total, streaming_used, streaming_total)."""
        base = _OFF_GPUS + gpu_idx * _GPU_SNAP_SIZE
        addr = self._addr
        o_su = base + GpuSnapshot.expert_stable_used.offset
        o_st = base + GpuSnapshot.expert_stable_total.offset
        o_ru = base + GpuSnapshot.expert_streaming_used.offset
        o_rt = base + GpuSnapshot.expert_streaming_total.offset

        def _read(_tx: SnapshotTransaction) -> tuple[int, int, int, int]:
            return (
                ctypes.c_uint32.from_address(addr + o_su).value,
                ctypes.c_uint32.from_address(addr + o_st).value,
                ctypes.c_uint32.from_address(addr + o_ru).value,
                ctypes.c_uint32.from_address(addr + o_rt).value,
            )

        return read_with_retry(addr, _read, self._policy)

    def inflight_transfers(self, gpu_idx: int) -> tuple[int, int]:
        """(h2d_count, d2h_count) for one GPU."""
        base = _OFF_GPUS + gpu_idx * _GPU_SNAP_SIZE
        addr = self._addr
        o_h2d = base + GpuSnapshot.inflight_h2d_count.offset
        o_d2h = base + GpuSnapshot.inflight_d2h_count.offset

        def _read(_tx: SnapshotTransaction) -> tuple[int, int]:
            return (
                ctypes.c_uint32.from_address(addr + o_h2d).value,
                ctypes.c_uint32.from_address(addr + o_d2h).value,
            )

        return read_with_retry(addr, _read, self._policy)

    # ── Expert residency ──────────────────────────────────────────────────

    def is_resident(self, layer: int, expert: int, gpu: int) -> bool:
        """Check if expert (layer, expert) is fully ready (HOT) on gpu."""
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu
        byte_off = _OFF_RESIDENCY + flat // 8
        bit = flat % 8
        val = self._tx().read_u8(byte_off)
        return bool(val & (1 << bit))

    # ── ELM tier state (ELM-8) ─────────────────────────────────────────────

    def expert_gpu_tier(self, layer: int, expert: int, gpu: int) -> int:
        """GPU-tier enum for (layer, expert, gpu). See GPU_TIER_* constants."""
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu
        return self._tx().read_u8(_OFF_GPU_TIER + flat)

    def expert_interest_count(self, layer: int, expert: int, gpu: int) -> int:
        """Active interest (reference) count for (layer, expert) on gpu."""
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu
        return self._tx().read_u8(_OFF_INTEREST + flat)

    def expert_host_tier(self, layer: int, expert: int) -> int:
        """Host-tier enum for (layer, expert). See HOST_TIER_* constants."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_u8(_OFF_HOST_TIER + idx)

    def is_host_warm(self, layer: int, expert: int) -> bool:
        """Fast bitmap check: is expert warm in host RAM?

        ``layer`` is MOE-RELATIVE (0 = first MoE layer) — the C++ ELM
        writes every per-expert snapshot array at
        ``(layer_idx - first_moe_layer) * kMaxExperts + expert``
        (expert_lifecycle_manager.cpp publish_host_state).  Layers >=
        MAX_MOE_LAYERS are not tracked (fail-closed False).
        """
        if not (0 <= layer < MAX_MOE_LAYERS and 0 <= expert < MAX_EXPERTS):
            return False
        bit_idx = layer * MAX_EXPERTS + expert
        byte_off = _OFF_HOST_BITMAP + bit_idx // 8
        bit = bit_idx % 8
        val = self._tx().read_u8(byte_off)
        return bool(val & (1 << bit))

    def host_warm_bitmap(self) -> bytes:
        """One-transaction copy of the whole host-resident bitmap (#91).

        Bit index = moe_relative_layer * MAX_EXPERTS + expert (LSB-first
        within each byte, matching the C++ ``1u << (idx % 8)`` writer).
        A single seqlock transaction replaces MAX_MOE_LAYERS×MAX_EXPERTS
        per-field transactions — the per-cycle full scan in
        ``_compute_host_resident_keys`` MUST use this (a per-bit scan costs
        ~seconds per cycle against a live daemon, INV-IPC-PUBLISH-THROTTLE).
        """
        addr = self._addr
        size = MAX_MOE_LAYERS * MAX_EXPERTS // 8

        def _read(_tx: SnapshotTransaction) -> bytes:
            return bytes(
                (ctypes.c_uint8 * size).from_address(addr + _OFF_HOST_BITMAP)
            )

        return read_with_retry(addr, _read, self._policy)

    def expert_last_change_ns(self, layer: int, expert: int) -> int:
        """Nanosecond timestamp of last ELM state change for (layer, expert)."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_u64(_OFF_LAST_CHANGE_NS + idx * 8)

    # ── Expert statistics ─────────────────────────────────────────────────

    def expert_frequency(self, layer: int, expert: int) -> float:
        """EWMA activation frequency for (layer, expert). Normalized [0,1]."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_f32(_OFF_FREQ + idx * 4)

    def expert_recency(self, layer: int, expert: int) -> float:
        """Normalized recency for (layer, expert). [0,1]."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_f32(_OFF_RECENCY + idx * 4)

    def expert_routing_weight(self, layer: int, expert: int) -> float:
        """Mean routing weight for (layer, expert). Normalized [0,1]."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_f32(_OFF_ROUTING_WT + idx * 4)

    def expert_autocorrelation(self, layer: int, expert: int) -> float:
        """Temporal autocorrelation for (layer, expert). [0,1]."""
        idx = layer * MAX_EXPERTS + expert
        return self._tx().read_f32(_OFF_AUTOCORR + idx * 4)

    def expert_stats(self, layer: int, expert: int
                     ) -> tuple[float, float, float, float]:
        """(frequency, recency, routing_weight, autocorrelation) in one transaction."""
        idx = layer * MAX_EXPERTS + expert
        addr = self._addr
        off_f = _OFF_FREQ + idx * 4
        off_r = _OFF_RECENCY + idx * 4
        off_w = _OFF_ROUTING_WT + idx * 4
        off_a = _OFF_AUTOCORR + idx * 4

        def _read(_tx: SnapshotTransaction) -> tuple[float, float, float, float]:
            return (
                ctypes.c_float.from_address(addr + off_f).value,
                ctypes.c_float.from_address(addr + off_r).value,
                ctypes.c_float.from_address(addr + off_w).value,
                ctypes.c_float.from_address(addr + off_a).value,
            )

        return read_with_retry(addr, _read, self._policy)

    def expert_stats_batch(
        self, keys: list[tuple[int, int]],
    ) -> list[tuple[float, float, float, float]]:
        """Batch read (frequency, recency, routing_weight, autocorr) for multiple experts.

        Reads all four stat arrays once, then indexes into them for each
        (layer, expert) pair.  Much faster than N individual expert_stats()
        calls when N > ~4, since it avoids N separate seqlock transactions.
        """
        addr = self._addr
        n = MAX_MOE_LAYERS * MAX_EXPERTS

        # TD-ORCH-PLAN-CRAWL: numpy end-to-end. The previous
        # read_f32_array path materialized 4 × 32768-entry Python float
        # lists PER CALL (this runs every Phase-4 PLAN cycle) — the
        # dominant term of the collector crawl. read_f32_np keeps the
        # same one-memcpy-per-array seqlock discipline.
        def _read(_tx: SnapshotTransaction):
            freq_all = _tx.read_f32_np(_OFF_FREQ, n)
            rec_all = _tx.read_f32_np(_OFF_RECENCY, n)
            rw_all = _tx.read_f32_np(_OFF_ROUTING_WT, n)
            ac_all = _tx.read_f32_np(_OFF_AUTOCORR, n)
            return freq_all, rec_all, rw_all, ac_all

        freq_all, rec_all, rw_all, ac_all = read_with_retry(
            addr, _read, self._policy,
        )

        import numpy as np
        layers = np.fromiter((k[0] for k in keys), np.int64, count=len(keys))
        experts = np.fromiter((k[1] for k in keys), np.int64, count=len(keys))
        idx = layers * MAX_EXPERTS + experts
        # layer is MOE-RELATIVE; out-of-range (negative = dense layer,
        # >= MAX_MOE_LAYERS = untracked tail) reads zero stats.
        valid = (layers >= 0) & (idx >= 0) & (idx < n)
        safe_idx = np.where(valid, idx, 0)
        cols = np.stack([
            np.where(valid, a[safe_idx], 0.0)
            for a in (freq_all, rec_all, rw_all, ac_all)
        ], axis=1).astype(float)
        return [tuple(row) for row in cols.tolist()]

    def all_expert_frequency(self, num_layers: int) -> list[float]:
        """All expert frequencies as flat list[num_layers * MAX_EXPERTS]."""
        count = num_layers * MAX_EXPERTS
        return self._tx().read_f32_array(_OFF_FREQ, count)

    def all_expert_recency(self, num_layers: int) -> list[float]:
        """All expert recencies as flat list[num_layers * MAX_EXPERTS]."""
        count = num_layers * MAX_EXPERTS
        return self._tx().read_f32_array(_OFF_RECENCY, count)

    def all_expert_routing_weight(self, num_layers: int) -> list[float]:
        """All expert routing weights as flat list[num_layers * MAX_EXPERTS]."""
        count = num_layers * MAX_EXPERTS
        return self._tx().read_f32_array(_OFF_ROUTING_WT, count)

    def all_expert_autocorrelation(self, num_layers: int) -> list[float]:
        """All expert autocorrelations as flat list[num_layers * MAX_EXPERTS]."""
        count = num_layers * MAX_EXPERTS
        return self._tx().read_f32_array(_OFF_AUTOCORR, count)

    # ── Acceptance rates ──────────────────────────────────────────────────

    def global_acceptance_rate(self) -> float:
        """Global EMA acceptance rate [0,1]."""
        return self._tx().read_f64(_OFF_GLOBAL_ACC)

    def windowed_acceptance_rate(self) -> float:
        """Recent-window acceptance rate [0,1]."""
        return self._tx().read_f64(_OFF_WINDOWED_ACC)

    def layer_skip_acceptance_rate(self) -> float:
        """Layer-skip acceptance EMA [0,1]. -1.0 if no layer-skip data."""
        return self._tx().read_f64(_OFF_LAYERSKIP_ACC)

    def total_verifications(self) -> int:
        """Lifetime verification count."""
        return self._tx().read_u64(_OFF_TOTAL_VERIF)

    def cumulative_acceptance(self) -> tuple[int, int]:
        """(total_accepted_tokens, total_attempted_tokens) in one transaction."""
        addr = self._addr

        def _read(_tx: SnapshotTransaction) -> tuple[int, int]:
            a = ctypes.c_uint64.from_address(addr + _OFF_TOTAL_ACCEPTED).value
            t = ctypes.c_uint64.from_address(addr + _OFF_TOTAL_ATTEMPTED).value
            return (a, t)

        return read_with_retry(addr, _read, self._policy)

    def per_request_acceptance(self) -> list[tuple[int, float]]:
        """Per-request acceptance entries: [(request_id, rate), ...]."""
        addr = self._addr

        def _read(_tx: SnapshotTransaction) -> list[tuple[int, float]]:
            n = ctypes.c_uint32.from_address(addr + _OFF_NUM_TRACKED).value
            n = min(n, MAX_TRACKED_REQUESTS)
            result: list[tuple[int, float]] = []
            for i in range(n):
                base = _OFF_PER_REQ + i * _REQ_ACCEPT_SIZE
                rid = ctypes.c_uint64.from_address(
                    addr + base + RequestAcceptance.request_id.offset).value
                rate = ctypes.c_double.from_address(
                    addr + base + RequestAcceptance.acceptance_rate.offset).value
                result.append((rid, rate))
            return result

        return read_with_retry(addr, _read, self._policy)

    # ── Workload detector ─────────────────────────────────────────────────

    def shift_detected(self) -> bool:
        """True if a workload shift was detected (one-shot flag)."""
        return self._tx().read_u8(_OFF_SHIFT) != 0

    # ── Transfer engine ───────────────────────────────────────────────────

    def total_inflight_transfers(self) -> int:
        """Global count of pending DMA operations."""
        return self._tx().read_u32(_OFF_INFLIGHT)

    # ── Host NUMA placement ──────────────────────────────────────────────

    def host_numa_node(self, layer: int, expert: int) -> int:
        """NUMA node of expert's warm cache buffer. -1 if not warm."""
        idx = layer * MAX_EXPERTS + expert
        val = self._tx().read_u8(_OFF_HOST_NUMA + idx)
        return val if val < 128 else val - 256

    # ── Per-NUMA host tier (ELM-8b) ────────────────────────────────────

    def host_numa_tier(self, layer: int, expert: int, numa: int) -> int:
        """Per-NUMA host tier for (layer, expert, numa). See HOST_TIER_* constants."""
        idx = layer * MAX_EXPERTS * MAX_NUMA + expert * MAX_NUMA + numa
        return self._tx().read_u8(_OFF_HOST_NUMA_TIER + idx)

    def host_numa_tiers(self, layer: int, expert: int) -> list[int]:
        """All NUMA tiers for (layer, expert). Returns list of MAX_NUMA HostTier values."""
        base = _OFF_HOST_NUMA_TIER + layer * MAX_EXPERTS * MAX_NUMA + expert * MAX_NUMA
        addr = self._addr

        def _read(tx: "SnapshotTransaction") -> list[int]:
            return [tx.read_u8(base + n) for n in range(MAX_NUMA)]

        return read_with_retry(addr, _read, self._policy)
