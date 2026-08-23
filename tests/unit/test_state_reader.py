"""Unit tests for state_reader.py.

Pure Python — no engine required. Allocates a ctypes StateSnapshot buffer,
writes known values under manual seqlock protocol, reads back via StateReader.
"""

import ctypes
import struct

import pytest

from orchestrator.shm_protocol import (
    GPU_TIER_ABSENT,
    GPU_TIER_HOT,
    GPU_TIER_TRANSFERRING,
    GpuSnapshot,
    HOST_TIER_COLD,
    HOST_TIER_WARM,
    MAX_EXPERTS,
    MAX_GPUS,
    MAX_MOE_LAYERS,
    MAX_NUMA,
    MAX_TRACKED_REQUESTS,
    RequestAcceptance,
    StateSnapshot,
)
from orchestrator.spsc_ring import RetryPolicy
from orchestrator.state_reader import StateReader

_SNAP_SIZE = ctypes.sizeof(StateSnapshot)

# Fast retry policy for tests (no real contention).
_TEST_POLICY = RetryPolicy(max_retries=100, stage_size=10, stage0_yield_every=5)


@pytest.fixture
def snap_env():
    """Allocate a zeroed StateSnapshot buffer.  Returns (buf, addr, snap, reader)."""
    buf = ctypes.create_string_buffer(_SNAP_SIZE)
    addr = ctypes.addressof(buf)
    snap = StateSnapshot.from_address(addr)
    snap.seqlock = 0  # Even = readable
    reader = StateReader(addr, retry_policy=_TEST_POLICY)
    return buf, addr, snap, reader


def _seqlock_write(snap: StateSnapshot, fn) -> None:
    """Write to snap under manual seqlock protocol."""
    snap.seqlock += 1   # Odd = writing
    fn(snap)
    snap.seqlock += 1   # Even = readable


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamps
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestamps:
    def test_cycle_count(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "daemon_cycle_count", 12345))
        assert reader.cycle_count() == 12345

    def test_timestamp_ns(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "timestamp_ns", 9999999999))
        assert reader.timestamp_ns() == 9999999999


# ═══════════════════════════════════════════════════════════════════════════════
# GPU metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestGpuMetrics:
    def test_num_gpus(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "num_gpus", 4))
        assert reader.num_gpus() == 4

    def test_gpu_snapshot(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.num_gpus = 2
            s.gpus[0].vram_used_bytes = 1000
            s.gpus[0].vram_total_bytes = 2000
            s.gpus[0].expert_stable_used = 10
            s.gpus[0].expert_stable_total = 50
            s.gpus[0].inflight_h2d_count = 3
            s.gpus[0].inflight_d2h_count = 1

        _seqlock_write(snap, _write)
        gs = reader.gpu_snapshot(0)
        assert gs.vram_used_bytes == 1000
        assert gs.vram_total_bytes == 2000
        assert gs.expert_stable_used == 10
        assert gs.inflight_h2d_count == 3

    def test_vram_usage(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.gpus[1].vram_used_bytes = 500
            s.gpus[1].vram_total_bytes = 1000

        _seqlock_write(snap, _write)
        used, total = reader.vram_usage(1)
        assert used == 500
        assert total == 1000

    def test_cache_fill(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.gpus[0].expert_stable_used = 5
            s.gpus[0].expert_stable_total = 20
            s.gpus[0].expert_streaming_used = 3
            s.gpus[0].expert_streaming_total = 10

        _seqlock_write(snap, _write)
        su, st, ru, rt = reader.cache_fill(0)
        assert (su, st, ru, rt) == (5, 20, 3, 10)

    def test_inflight_transfers_per_gpu(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.gpus[2].inflight_h2d_count = 7
            s.gpus[2].inflight_d2h_count = 2

        _seqlock_write(snap, _write)
        h2d, d2h = reader.inflight_transfers(2)
        assert h2d == 7
        assert d2h == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Expert residency
# ═══════════════════════════════════════════════════════════════════════════════

class TestResidency:
    def test_is_resident(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert, gpu = 2, 5, 1
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu

        def _write(s):
            byte_idx = flat // 8
            bit = flat % 8
            s.residency_bitmap[byte_idx] |= (1 << bit)

        _seqlock_write(snap, _write)
        assert reader.is_resident(layer, expert, gpu) is True
        assert reader.is_resident(layer, expert, 0) is False  # Different GPU

    def test_expert_gpu_tier(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert, gpu = 1, 3, 0
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu

        def _write(s):
            s.expert_gpu_tier[flat] = GPU_TIER_HOT

        _seqlock_write(snap, _write)
        assert reader.expert_gpu_tier(layer, expert, gpu) == GPU_TIER_HOT

    def test_expert_interest_count(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert, gpu = 0, 5, 1
        flat = layer * MAX_EXPERTS * MAX_GPUS + expert * MAX_GPUS + gpu

        def _write(s):
            s.expert_interest_count[flat] = 3

        _seqlock_write(snap, _write)
        assert reader.expert_interest_count(layer, expert, gpu) == 3

    def test_expert_host_tier(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 2, 10
        idx = layer * MAX_EXPERTS + expert

        def _write(s):
            s.host_tier[idx] = HOST_TIER_WARM

        _seqlock_write(snap, _write)
        assert reader.expert_host_tier(layer, expert) == HOST_TIER_WARM

    def test_is_host_warm(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 0, 0
        bit_idx = layer * MAX_EXPERTS + expert

        def _write(s):
            s.host_resident_bitmap[bit_idx // 8] |= (1 << (bit_idx % 8))

        _seqlock_write(snap, _write)
        assert reader.is_host_warm(layer, expert) is True
        assert reader.is_host_warm(layer, expert + 1) is False
        # Out-of-range layers are untracked → fail-closed False (#91,
        # INV-IPC-MOE-REL: layer is MOE-relative, >= MAX_MOE_LAYERS or
        # negative must never index past the bitmap).
        assert reader.is_host_warm(MAX_MOE_LAYERS, expert) is False
        assert reader.is_host_warm(-1, expert) is False

    def test_host_warm_bitmap_batch(self, snap_env):
        """#91: one-transaction whole-bitmap read matches per-bit reads.

        Bit index = moe_relative_layer * MAX_EXPERTS + expert, LSB-first
        (the C++ ELM writer's `1u << (idx % 8)`).
        """
        import numpy as np

        buf, addr, snap, reader = snap_env
        warm = [(0, 0), (2, 7), (MAX_MOE_LAYERS - 1, MAX_EXPERTS - 1)]

        def _write(s):
            for layer, expert in warm:
                bit_idx = layer * MAX_EXPERTS + expert
                s.host_resident_bitmap[bit_idx // 8] |= (1 << (bit_idx % 8))

        _seqlock_write(snap, _write)
        bitmap = reader.host_warm_bitmap()
        assert len(bitmap) == MAX_MOE_LAYERS * MAX_EXPERTS // 8
        bits = np.unpackbits(
            np.frombuffer(bitmap, dtype=np.uint8), bitorder="little",
        )
        set_idx = {int(i) for i in np.nonzero(bits)[0]}
        assert set_idx == {
            layer * MAX_EXPERTS + expert for layer, expert in warm
        }
        # Consistency with the per-bit reader.
        for layer, expert in warm:
            assert reader.is_host_warm(layer, expert) is True

    def test_expert_last_change_ns(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 1, 7
        idx = layer * MAX_EXPERTS + expert

        def _write(s):
            s.expert_last_change_ns[idx] = 123456789012345

        _seqlock_write(snap, _write)
        assert reader.expert_last_change_ns(layer, expert) == 123456789012345

    def test_gpu_tier_constants(self, snap_env):
        assert GPU_TIER_ABSENT == 0
        assert GPU_TIER_TRANSFERRING == 2
        assert GPU_TIER_HOT == 6

    def test_host_tier_constants(self, snap_env):
        assert HOST_TIER_COLD == 0
        assert HOST_TIER_WARM == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Expert statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpertStats:
    def test_expert_frequency(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 3, 7
        idx = layer * MAX_EXPERTS + expert

        def _write(s):
            s.expert_frequency[idx] = 0.75

        _seqlock_write(snap, _write)
        assert abs(reader.expert_frequency(layer, expert) - 0.75) < 1e-6

    def test_expert_recency(self, snap_env):
        buf, addr, snap, reader = snap_env
        idx = 1 * MAX_EXPERTS + 2

        def _write(s):
            s.expert_recency[idx] = 0.33

        _seqlock_write(snap, _write)
        assert abs(reader.expert_recency(1, 2) - 0.33) < 1e-6

    def test_expert_routing_weight(self, snap_env):
        buf, addr, snap, reader = snap_env
        idx = 0 * MAX_EXPERTS + 0

        def _write(s):
            s.expert_routing_weight[idx] = 0.99

        _seqlock_write(snap, _write)
        assert abs(reader.expert_routing_weight(0, 0) - 0.99) < 1e-6

    def test_expert_autocorrelation(self, snap_env):
        buf, addr, snap, reader = snap_env
        idx = 2 * MAX_EXPERTS + 100

        def _write(s):
            s.expert_temporal_autocorr[idx] = 0.42

        _seqlock_write(snap, _write)
        assert abs(reader.expert_autocorrelation(2, 100) - 0.42) < 1e-6

    def test_expert_stats_multi_field(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 1, 5
        idx = layer * MAX_EXPERTS + expert

        def _write(s):
            s.expert_frequency[idx] = 0.1
            s.expert_recency[idx] = 0.2
            s.expert_routing_weight[idx] = 0.3
            s.expert_temporal_autocorr[idx] = 0.4

        _seqlock_write(snap, _write)
        f, r, w, a = reader.expert_stats(layer, expert)
        assert abs(f - 0.1) < 1e-6
        assert abs(r - 0.2) < 1e-6
        assert abs(w - 0.3) < 1e-6
        assert abs(a - 0.4) < 1e-6

    def test_all_expert_frequency(self, snap_env):
        buf, addr, snap, reader = snap_env
        num_layers = 3

        def _write(s):
            for l in range(num_layers):
                for e in range(MAX_EXPERTS):
                    s.expert_frequency[l * MAX_EXPERTS + e] = float(l * MAX_EXPERTS + e) / 1000.0

        _seqlock_write(snap, _write)
        result = reader.all_expert_frequency(num_layers)
        assert len(result) == num_layers * MAX_EXPERTS
        # Spot-check
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1 * MAX_EXPERTS + 5] - (1 * MAX_EXPERTS + 5) / 1000.0) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# Acceptance rates
# ═══════════════════════════════════════════════════════════════════════════════

class TestAcceptance:
    def test_global_acceptance_rate(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "global_acceptance_rate", 0.85))
        assert abs(reader.global_acceptance_rate() - 0.85) < 1e-10

    def test_windowed_acceptance_rate(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "windowed_acceptance_rate", 0.72))
        assert abs(reader.windowed_acceptance_rate() - 0.72) < 1e-10

    def test_layer_skip_acceptance_rate(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "layer_skip_acceptance_rate", -1.0))
        assert reader.layer_skip_acceptance_rate() == -1.0

    def test_total_verifications(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "total_verifications", 42))
        assert reader.total_verifications() == 42

    def test_cumulative_acceptance(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.total_accepted_tokens = 100
            s.total_attempted_tokens = 150

        _seqlock_write(snap, _write)
        accepted, attempted = reader.cumulative_acceptance()
        assert accepted == 100
        assert attempted == 150

    def test_per_request_acceptance_empty(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "num_tracked_requests", 0))
        assert reader.per_request_acceptance() == []

    def test_per_request_acceptance_populated(self, snap_env):
        buf, addr, snap, reader = snap_env

        def _write(s):
            s.per_request_acceptance[0].request_id = 100
            s.per_request_acceptance[0].acceptance_rate = 0.75
            s.per_request_acceptance[1].request_id = 200
            s.per_request_acceptance[1].acceptance_rate = 0.50
            s.num_tracked_requests = 2

        _seqlock_write(snap, _write)
        result = reader.per_request_acceptance()
        assert len(result) == 2
        ids = {r[0] for r in result}
        assert 100 in ids
        assert 200 in ids
        # Check rates
        for rid, rate in result:
            if rid == 100:
                assert abs(rate - 0.75) < 1e-10
            elif rid == 200:
                assert abs(rate - 0.50) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# Workload detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkload:
    def test_shift_detected_false(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "shift_detected", 0))
        assert reader.shift_detected() is False

    def test_shift_detected_true(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "shift_detected", 1))
        assert reader.shift_detected() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransfers:
    def test_total_inflight_transfers(self, snap_env):
        buf, addr, snap, reader = snap_env
        _seqlock_write(snap, lambda s: setattr(s, "total_inflight_transfers", 17))
        assert reader.total_inflight_transfers() == 17


# ═══════════════════════════════════════════════════════════════════════════════
# Host NUMA placement
# ═══════════════════════════════════════════════════════════════════════════════

class TestHostNuma:
    def test_host_numa_default_minus_one(self, snap_env):
        buf, addr, snap, reader = snap_env
        idx = 2 * MAX_EXPERTS + 10
        def _write(s):
            s.expert_host_numa[idx] = -1
        _seqlock_write(snap, _write)
        assert reader.host_numa_node(2, 10) == -1

    def test_host_numa_node_value(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 1, 5
        idx = layer * MAX_EXPERTS + expert
        def _write(s):
            s.expert_host_numa[idx] = 2
        _seqlock_write(snap, _write)
        assert reader.host_numa_node(layer, expert) == 2

    def test_host_numa_node_zero(self, snap_env):
        buf, addr, snap, reader = snap_env
        idx = 0 * MAX_EXPERTS + 0
        def _write(s):
            s.expert_host_numa[idx] = 0
        _seqlock_write(snap, _write)
        assert reader.host_numa_node(0, 0) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Per-NUMA host tier (ELM-8b)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHostNumaTier:
    def test_host_numa_tier_default_cold(self, snap_env):
        buf, addr, snap, reader = snap_env
        # All zeros by default = HOST_TIER_COLD.
        assert reader.host_numa_tier(0, 0, 0) == HOST_TIER_COLD

    def test_host_numa_tier_set_warm(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert, numa = 1, 5, 3
        idx = layer * MAX_EXPERTS * MAX_NUMA + expert * MAX_NUMA + numa
        def _write(s):
            s.host_numa_tier[idx] = HOST_TIER_WARM
        _seqlock_write(snap, _write)
        assert reader.host_numa_tier(layer, expert, numa) == HOST_TIER_WARM
        # Other NUMA nodes still cold.
        assert reader.host_numa_tier(layer, expert, 0) == HOST_TIER_COLD

    def test_host_numa_tiers_all(self, snap_env):
        buf, addr, snap, reader = snap_env
        layer, expert = 2, 10
        base = layer * MAX_EXPERTS * MAX_NUMA + expert * MAX_NUMA
        def _write(s):
            s.host_numa_tier[base + 0] = HOST_TIER_WARM
            s.host_numa_tier[base + 2] = HOST_TIER_WARM
        _seqlock_write(snap, _write)
        tiers = reader.host_numa_tiers(layer, expert)
        assert len(tiers) == MAX_NUMA
        assert tiers[0] == HOST_TIER_WARM
        assert tiers[1] == HOST_TIER_COLD
        assert tiers[2] == HOST_TIER_WARM
        for n in range(3, MAX_NUMA):
            assert tiers[n] == HOST_TIER_COLD

    def test_host_numa_tier_different_experts_independent(self, snap_env):
        buf, addr, snap, reader = snap_env
        base0 = 0 * MAX_EXPERTS * MAX_NUMA + 0 * MAX_NUMA
        base1 = 0 * MAX_EXPERTS * MAX_NUMA + 1 * MAX_NUMA
        def _write(s):
            s.host_numa_tier[base0 + 0] = HOST_TIER_WARM
            s.host_numa_tier[base1 + 3] = HOST_TIER_WARM
        _seqlock_write(snap, _write)
        # Expert 0 warm on NUMA 0 only.
        assert reader.host_numa_tier(0, 0, 0) == HOST_TIER_WARM
        assert reader.host_numa_tier(0, 0, 3) == HOST_TIER_COLD
        # Expert 1 warm on NUMA 3 only.
        assert reader.host_numa_tier(0, 1, 0) == HOST_TIER_COLD
        assert reader.host_numa_tier(0, 1, 3) == HOST_TIER_WARM


# ═══════════════════════════════════════════════════════════════════════════════
# Seqlock conflict detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeqlock:
    def test_odd_seqlock_retries(self, snap_env):
        """Verify reads eventually succeed after seqlock transitions odd->even."""
        buf, addr, snap, reader = snap_env
        # Set seqlock to odd (writer active).
        snap.seqlock = 1
        snap.daemon_cycle_count = 999

        # Then set it to even (writer done) — simulates writer finishing.
        snap.seqlock = 2

        # Reader should succeed (seqlock is now even=2).
        assert reader.cycle_count() == 999
